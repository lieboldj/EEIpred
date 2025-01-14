from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from glinter.model.atomgcn import AtomGCN
# from glinter.module.conv import make_layer, ResNet, BasicBlock2d, Bottleneck2d
from glinter.esm_embed import load_esm_model
from glinter.modules.atomgcn import AtomGCN
from glinter.modules.conv import make_layer, ResNet, BasicBlock2d, Bottleneck2d
from sklearn.metrics import roc_auc_score

# copied from esm
def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def build_eval_str_list(sep=',', cast=float):
    def _eval_str_list(s):
        return [ cast(_) for _ in s.split(sep) ]
    return _eval_str_list

class MSAModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--node-embed-dim', type=int, default=43,)
        parser.add_argument('--num-1d-layers', type=int, default=1,)
        parser.add_argument('--rates', type=build_eval_str_list(), default=[0.5],)
        parser.add_argument('--rs', type=build_eval_str_list(), default=[12],)
        parser.add_argument(
            '--row-attn-op', type=str, choices=[
                'lower_tri', 'upper_tri', 'sym', 'apc',
            ],
            default='sym',
        )

    def __init__(
        self, args, esm_embed=None, prepend_bos=False, gen_esm=False
    ):
        super().__init__()
        self.args = args
        
        self._gen_esm = gen_esm
        if self._gen_esm:
            assert esm_embed is not None

        self.esm_embed = None
        self.prepend_bos = prepend_bos

        embed_dim = 0
        if 'esm' in args.feature:
            assert esm_embed is not None
            self.esm_embed = esm_embed
            if not self._gen_esm:
                embed_dim += 144

        elif 'pickled-esm' in args.feature:
            embed_dim += 144

        if 'ccm' in args.feature:
            embed_dim += 1 # using ccm instead of msa embeddings

        encoder_1d, _encoder_1d_dim = self._build_encoder_1d()
        self.encoder_1d = encoder_1d
        embed_dim += _encoder_1d_dim * 2

        if not self._gen_esm:
            _conv1 = make_layer(BasicBlock2d, embed_dim, 96, 16)
            self.resnet = ResNet([_conv1,]) 
            self.fc = nn.Conv2d(96, 2, kernel_size=1)


    def _build_encoder_1d(self,):
        encoder_1d = None
        embed_dim = self.args.node_embed_dim
        output_dim = 0
        num_layers = self.args.num_1d_layers
        src_graphs = []

        if 'ca-embed' in self.args.feature:
            _local_dim = 128
            output_dim = 128
            encoder_1d = nn.Sequential(
                nn.Conv1d(embed_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, output_dim, 5, padding=2),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )

        if self.args.feature.use('coordinate-ca-graph','distance-ca-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=embed_dim,
                    use_pos=self.args.feature.use('coordinate-ca-graph'),
                    tgt_dim=_local_dim,
                    use_concat=True,
                    edge_dim=(
                        1 if self.args.feature.use('distance-ca-graph') else 0
                    ),
                ),
            )

        if self.args.feature.use('atom-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=33,
                    use_pos=True,
                    tgt_dim=_local_dim,
                    use_concat=True,
                    edge_dim=1,
                ),
            )

        if self.args.feature.use('surface-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=0,
                    use_nor=True,
                    use_pos=True,
                    use_concat=True,
                    tgt_dim=_local_dim,
                ),
            )

        if len(src_graphs) > 0:
            if num_layers > 1:
                assert num_layers - 1 == len(self.args.rates)
                assert num_layers - 1 == len(self.args.rs)
                ks = [ -1 ] * (num_layers - 1)
                sa_dims = [ _local_dim ] * (num_layers - 1)
                fp_dims = [ _local_dim ] * (num_layers - 1)
            else:
                ks = None
                sa_dims = None
                fp_dims = None

            output_dim = 128
            encoder_1d = AtomGCN(
                embed_dim, output_dim, tuple(src_graphs), num_sa=num_layers-1, 
                use_fp=True, rates=self.args.rates, rs=self.args.rs, ks=ks,
                sa_dims=sa_dims, fp_dims=fp_dims,
            )

        return encoder_1d, output_dim

    def forward(self, data):
        x = None
        if self.esm_embed is not None:
            try:
                with torch.no_grad():
                    self.esm_embed.eval()
                    msa = data['msa']
                    x = self.esm_embed(msa)['row_attentions']
                
                if self.prepend_bos:
                    x = x[..., 1:, 1:]

                B, L, N, K, K = x.size()
                x = x.view(B, L*N, K, K)

                reclen = int(data['reclen'])
                liglen = int(data['liglen'])

                _op = self.args.row_attn_op
    
                if _op == 'lower_tri':
                    x = x[:, :, :reclen, reclen:]
                elif _op == 'upper_tri':
                    x = x[:, :, reclen:, :reclen].transpose(-2,-1)
                elif _op == 'sym':
                    x = (
                        x[:, :, :reclen, reclen:] + 
                        x[:, :, reclen:, :reclen].transpose(-2,-1)
                    )
                elif _op == 'apc': # sym, then apc
                    x = apc(
                        x + x.transpose(-2,-1)
                    )[:, :, :reclen, reclen:]
                else:
                    raise ValueError()

                if x.size(-1) != int(liglen):
                    raise RuntimeError(
                        'shape mismatch in concated msa: '
                        f'{x.size(), msa.size(), reclen, liglen}'
                    )

            except RuntimeError as e:
                raise e

            if self._gen_esm:
                assert B == 1
                x = x.squeeze(0)
                return x

        if 'pickled-esm' in self.args.feature:
            assert x is None
            x = data['esm']
 
        if 'ccm' in self.args.feature:
            y = data['ccm']
            if x is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y

        if self.encoder_1d is not None:
            y_rec, y_lig = self.encoder_1d_forward(data)
            y_rec = y_rec[:, :, data['recidx'][0]]
            y_lig = y_lig[:, :, data['ligidx'][0]]
            reclen, liglen = y_rec.size(-1), y_lig.size(-1)
            y = torch.cat(
                (
                    y_rec.unsqueeze(3).expand(-1, -1, -1, liglen),
                    y_lig.unsqueeze(2).expand(-1, -1, reclen, -1),
                ),
                dim=1,
            )
            if x is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y
        
        g = self.resnet(x)
        logits = self.fc(g)
        lprobs = F.log_softmax(logits, dim=1).permute(0,2,3,1)

        return lprobs

    def encoder_1d_forward(self, data):
        if 'ca-embed' in self.args.feature:
            y_rec = self.encoder_1d(data['rec_embed'])
            y_lig = self.encoder_1d(data['lig_embed'])
            assert y_rec.size(0) == 1 and y_lig.size(0) == 1

        rec_graphs, lig_graphs = [], []
        if self.args.feature.use('distance-ca-graph', 'coordinate-ca-graph'):
            rec_cag = data['rec_cag']
            lig_cag = data['lig_cag']
            rec_graphs.append(rec_cag)
            lig_graphs.append(lig_cag)
            if self.args.feature.use('atom-graph'):
                rec_graphs.append(data['rec_atg'])
                lig_graphs.append(data['lig_atg'])

            if self.args.feature.use('surface-graph'):
                rec_graphs.append(data['rec_sug'])
                lig_graphs.append(data['lig_sug'])

            y_rec = self.encoder_1d(
                rec_cag.x, rec_cag.pos, rec_cag.lrf, rec_graphs,
            )
            y_rec = y_rec.unsqueeze(0).transpose(1,2)
            y_lig = self.encoder_1d(
                lig_cag.x, lig_cag.pos, lig_cag.lrf, lig_graphs,
            )
            y_lig = y_lig.unsqueeze(0).transpose(1,2)

        return y_rec, y_lig

def move_to_cuda_(batch):
    for key in batch:
        value = batch[key]
        if hasattr(value, 'cuda'):
            batch[key] = value.cuda()
        elif isinstance(value, dict):
            move_to_cuda_(value)

def cut_msa_(batch, num_seq=128):
    batch['msa'] = batch['msa'][:, :num_seq]

def forward_backward_method(batch, labels, model, optimizer, criterion, correct_order, train_loss):
    labels_oh = labels.long().unsqueeze(0)
    labels_ohT = labels.T.long().unsqueeze(0)
    optimizer.zero_grad()
    output = {'model':{}}
    move_to_cuda_(batch['data'])
    output['model']['output'] = model(batch['data']).permute(0,3,1,2)

    if correct_order:
        loss = criterion(output['model']['output'], labels_oh)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    else:
        loss = criterion(output['model']['output'], labels_ohT)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

def forward_only_method(batch, labels, model, criterion, correct_order, val_loss):
    labels_oh = labels.long().unsqueeze(0)
    labels_ohT = labels.T.long().unsqueeze(0)
    output = {'model':{}}
    move_to_cuda_(batch['data'])
    output['model']['output'] = model(batch['data']).permute(0,3,1,2)

    if correct_order:
        loss = criterion(output['model']['output'], labels_oh)
        val_loss.append(loss.item())
    else:
        loss = criterion(output['model']['output'], labels_ohT)
        val_loss.append(loss.item())
    return val_loss, output['model']['output']

def dataset_check(dataset, i):
    try:
        t = dataset[i]
    except:
        print(i)
        return False
    # if first and second PDB ID chain is equal
    if dataset[i][2][0] == dataset[i][2][1]:
        #continue
        try:
            batch1 = collater([ dataset[i][0][0]])
            batch2 = collater([ dataset[i][0][0]])
        except:
            print("dataset_check", dataset[i][2][0], dataset[i][2][1])
            return False
        if batch1["data"]["esm"].shape[-2] == 1 or \
            batch1["data"]["esm"].shape[-1] == 1:
            #print(i, "shape = 1")
            return False
        #print(batch1, batch2)
    else: 
        try: 
            batch1 = collater([ dataset[i][0][0]])
            batch2 = collater([ dataset[i][0][1]])
        except:
            print("dataset_check", dataset[i][2][0], dataset[i][2][1])
            return False
        if batch1["data"]["esm"].shape[-2] == 1 or \
            batch1["data"]["esm"].shape[-1] == 1:
            #print(i, "shape = 1")
            return False
    return [True, batch1, batch2]

def get_top_k_prec(predicted_flat, labels_flat, precision_score, k = 10):

    k = k
    _, topk_indices = torch.topk(predicted_flat, k, dim=0)

    # Convert the labels to a boolean tensor indicating whether each element is positive (True) or negative (False)
    positive_labels = (labels_flat[:, 0] == 1)

    # Count the number of true positives in the top-k predictions
    true_positives = torch.sum(positive_labels[topk_indices[1:k+1]]).float()
    precision = true_positives / k

    # Calculate the precision for each sample

    precision_score.append(precision.item())
    return precision_score

def read_residue_positions(pos_file):
    if not os.path.exists(pos_file):
        return
    with open(pos_file, 'rt') as fh:
        pos = [ int(_) for _ in fh.readline().strip().split() ]
    pos = np.array(pos, dtype=np.int64)
    return pos

if __name__ == '__main__':
    
    import argparse
    from glinter.dataset.dimer_dataset import DimerDataset
    from glinter.dataset.exon_dataset import ProtPairDataset
    from glinter.dataset.collater import Collater
    from glinter.models.checkpoint_utils import load_state
    from tqdm import tqdm
    torch.autograd.set_detect_anomaly(True)
    # torch.manual_seed(123)
    parser = argparse.ArgumentParser()
    DimerDataset.add_args(parser) # needed to add args to parser for MSA Model
    MSAModel.add_args(parser)
    parser.add_argument('--ckpt-path', type=Path)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--generate-esm-attention', action='store_true')
    parser.add_argument('--cuda', type=str, default=1)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--data_root', type=str, default='')
    args, _ = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MSAModel(args)
    collater = Collater()

    # dataset args, list of training files, 
    dataset_train = ProtPairDataset(args, f"{args.data_root}{args.dataset}/train{args.fold}_glinter.txt") #_new
    dataset_val = ProtPairDataset(args, f"{args.data_root}{args.dataset}/val{args.fold}_glinter.txt")
    dataset_test = ProtPairDataset(args, f"{args.data_root}{args.dataset}/test{args.fold}_glinter.txt")


    # for the training background
    #dataset_test = ProtPairDataset(args, f"{args.data_root}{args.dataset}/train{args.fold}.txt")
    #dataset_test = ProtPairDataset(args, f"{args.dataset}val/val{args.fold}.txt")

    # weighted cross entropy loss with weight 5 for positive samples
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 5]).cuda())
    num_classes = 2
    weight_tensor = torch.ones(num_classes).cuda()
    weight_tensor[0] = 5 # set class for interaction to 5
    # use this loss because log_softmax is used in the model
    criterion = nn.NLLLoss(weight=weight_tensor)

    # Adam optimizer, ß2 = 0.9999 and lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,\
                                 betas = (0.9,0.9999))
    # learning rate scheduler, reduce half after 4 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                step_size=4, gamma=0.5)
    save_path = "ckpts/"

    epochs = 20
    counter = 0
    model = model.cuda()
    
    #print(dataset.dimers, args.dimer_root, args.esm_root)
    if args.training:
        best_val = 0
        for epoch in range(epochs):

            ###################
            # train the model #
            ###################

            train_loss = []
            model.train()
            dataset = dataset_train
            for i in tqdm(range(len(dataset))):
                if not dataset_check(dataset, i):
                    continue
                batch1, batch2 = dataset_check(dataset, i)[1:]

                labels = dataset[i][1].to(device)
                # switch labels so class 0 is interaction
                labels = torch.logical_not(labels)
                correct_order = ord(dataset[i][2][0]) <= ord(dataset[i][2][1])

                # call model for both, prot1/prot2 and prot2/prot1 and update weights each time
                train_loss = forward_backward_method(batch1, labels, model, optimizer, criterion, correct_order, train_loss)
                train_loss = forward_backward_method(batch2, labels, model, optimizer, criterion, not correct_order, train_loss)

            scheduler.step()
            #train_loss.append(loss1.item())
            #torch.save(model.state_dict(), save_path + f'{args.dataset}train{args.fold}/model_{args.fold}_{epoch}.pt')
            print(f"Epoch {epoch}, train loss: {np.mean(train_loss)}")

            ######################
            # validate the model #
            ######################

            model.eval()
            val_loss = []
            dataset = dataset_val
            precision_score = []

            for i in tqdm(range(len(dataset))):
                if not dataset_check(dataset, i):
                    continue
                with torch.no_grad(): 
                    batch1, batch2 = dataset_check(dataset, i)[1:]
                    labels = dataset[i][1].to(device)
                    # switch labels so class 0 is interaction
                    labels = torch.logical_not(labels)
                    
                    correct_order = ord(dataset[i][2][0]) <= ord(dataset[i][2][1])
                    val_loss, score1 = forward_only_method(batch1, labels, model, criterion, correct_order, val_loss)
                    val_loss, score2 = forward_only_method(batch2, labels, model, criterion, not correct_order, val_loss)
                    if correct_order:
                        score_1 = (torch.exp(score1[0,1,:,:])\
                            + torch.exp(score2[0,1,:,:].T))/2 
                        score_0 = (torch.exp(score1[0,0,:,:])\
                            + torch.exp(score2[0,0,:,:].T))/2
                        score = torch.stack((score_0.flatten(), score_1.flatten()), dim=1)
                        
                    else:
                        score_1 = (torch.exp(score1[0,1,:,:].T)\
                            + torch.exp(score2[0,1,:,:]))/2
                        score_0 = (torch.exp(score1[0,0,:,:].T)\
                            + torch.exp(score2[0,0,:,:]))/2
                        score = torch.stack((score_0.flatten(), score_1.flatten()), dim=1)
                        
                    labels_oh = torch.nn.functional.one_hot(labels.long().unsqueeze(0))
                    labels_flat = labels_oh.view(-1, 2)
                    precision_score = get_top_k_prec(score, labels_flat, precision_score, k = 10)

            print(f"Epoch {epoch}, val loss: {np.mean(val_loss)}")
            print(f"Epoch {epoch}, val precision: {np.mean(precision_score)}")
            if best_val < np.mean(precision_score):
                best_val = np.mean(precision_score)
                # check if folder exists
                if not os.path.exists(save_path + f"{args.dataset}/train{args.fold}"):
                    os.makedirs(save_path + f"{args.dataset}/train{args.fold}")
                torch.save(model.state_dict(), save_path + f'{args.dataset}/train{args.fold}/model_{args.fold}_best.pt')
                torch.save(model.state_dict(), save_path + f'{args.dataset}/train{args.fold}/model_{args.fold}_{epoch}.pt')
                print(f"Epoch {epoch}, best val precision: {np.mean(precision_score)}")

    ##################
    # test the model #
    ##################
    else:
        model.eval()
        #path_ckpt = Path(save_path + f"{args.dataset}/train{args.fold}/model_{args.fold}_{args.epoch}.pt")
        path_ckpt = Path(save_path + f"{args.dataset}/train{args.fold}/model_{args.fold}_best.pt")
        #path_ckpt = Path(save_path + "glinter1.pt")
        assert path_ckpt.exists(), f"{path_ckpt} does not exist"

        state = load_state(path_ckpt)
        model.load_state_dict(state, strict=True)
        dataset = dataset_test
        mode = "test"
        precision_score = []
        roc_scores = []
        for i in tqdm(range(len(dataset))):

            if not dataset_check(dataset, i):
                print(i)
                continue
            
            with torch.no_grad():
                batch1, batch2 = dataset_check(dataset, i)[1:]

                labels = dataset[i][1].to(device)
                # switch 0 and 1 in labels, so index 0 is interacting and 1 is not
                labels = torch.logical_not(labels)
                labels_oh = torch.nn.functional.one_hot(labels.long().unsqueeze(0))
                labels_flat = labels_oh.view(-1, 2)

                correct_order = ord(dataset[i][2][0]) <= ord(dataset[i][2][1])

                output1 = {'model':{}}
                output2 = {'model':{}}
            
                move_to_cuda_(batch1['data'])
                move_to_cuda_(batch2['data'])

                output1['model']['output'] = model(batch1['data']).permute(0,3,1,2)#.cpu()
                output2['model']['output'] = model(batch2['data']).permute(0,3,1,2)#.cpu()
                # order alphabetically to take correct pair
                ################################
                # change train /test path here #
                ################################
                if not os.path.exists(save_path + f"{args.dataset}/{mode}{args.fold}"):
                    os.makedirs(save_path + f"{args.dataset}/{mode}{args.fold}")

                if correct_order:
                    score_1 = (np.exp(output1['model']['output'][0,1,:,:].cpu().numpy())\
                        + np.exp(output2['model']['output'][0,1,:,:].cpu().numpy().T))/2 #.cpu.numpy())
                    score_0 = (np.exp(output1['model']['output'][0,0,:,:].cpu().numpy())\
                        + np.exp(output2['model']['output'][0,0,:,:].cpu().numpy().T))/2
                    np.save(save_path + f"{args.dataset}/{mode}{args.fold}/score_{str(args.esm_root)[13:]}.npy", score_0)
                    score = np.stack((score_0.flatten(), score_1.flatten()), axis=1)
                else:
                    score_1 = (np.exp(output1['model']['output'][0,1,:,:].cpu().numpy().T)\
                        + np.exp(output2['model']['output'][0,1,:,:].cpu().numpy()))/2
                    score_0 = (np.exp(output1['model']['output'][0,0,:,:].cpu().numpy().T)\
                        + np.exp(output2['model']['output'][0,0,:,:].cpu().numpy()))/2
                    np.save(save_path + f"{args.dataset}/{mode}{args.fold}/score_{str(args.esm_root)[13:]}.npy", score_0)
                    score = np.stack((score_0.flatten(), score_1.flatten()), axis=1)
                #print(score.shape, score_0.shape)
                #exit()
                # store score in numpy files for exon evaluation
                #np.save(save_path + f'{args.dataset}/test{args.fold}/{dataset[i][0].dimers[0][0]}.npy', score)

                name1 = str(args.esm_root)[13:19]
                name2 = str(args.esm_root)[20:]
                pos1 = read_residue_positions(f'examples/PDB/{name1}:{name2}/{name1}/{name1}.pos')
                pos2 = read_residue_positions(f'examples/PDB/{name1}:{name2}/{name2}/{name2}.pos')
                if pos1 is not None and pos2 is not None:
                    _pos1 = np.repeat(pos1[:,np.newaxis], len(pos2), axis=-1)
                    _pos2 = np.repeat(pos2[np.newaxis, :], len(pos1), axis=0)
                    #print(batch1['data']['recidx'].cpu().squeeze(0).numpy(), batch1['data']['ligidx'].cpu().squeeze(0).numpy())
                    ref_pos = np.concatenate(
                        (_pos1[...,np.newaxis], _pos2[...,np.newaxis]), axis=-1
                    )
                    top_idx = np.argsort(score_0.reshape(-1))[::-1]
                    ranked_score = score_0.reshape(-1)[top_idx]
                    ranked_pos_pair = ref_pos.reshape(-1, 2)[top_idx]
                    rank = []
                    for i in range(len(ranked_score)):
                        rec_pos, lig_pos = ranked_pos_pair[i]
                        rank.append((rec_pos, lig_pos, ranked_score[i]))

                    with open(f'examples/PDB/{str(args.esm_root)[13:]}/ranked_pairs.txt', 'wt') as fh:
                        fh.write('# seq1 seq2 prob\n')
                        for p1, p2, s in rank:
                            fh.write(f'{int(p1)} {int(p2)} {float(s):.4f}\n')
                            
                precision_score = get_top_k_prec(torch.tensor(score), labels_flat, precision_score, k = 10)

                try:    
                    roc_score = roc_auc_score(labels_flat[:,0].cpu(), torch.tensor(score)[:,0].cpu())
                    roc_scores.append(roc_score)
                except:
                    print("roc_prob")

        print(f"Precision score: {np.mean(precision_score)}")
        print(len(precision_score))
        print(f"ROC score: {np.mean(roc_scores)}")


