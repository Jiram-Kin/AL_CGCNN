#
import torch
import sys
sys.path.append('.')
import os
import pandas as pd
import numpy as np

from deepchem.models.torch_models import MPNNModel, CGCNNModel
import deepchem as dc
from deepchem.feat.material_featurizers import CGCNNFeaturizer
import pymatgen as mg
import pymatgen.core 
from pymatgen.io.cif import CifParser
from deepchem.feat.graph_data import GraphData
from AS_Molecule.single_model_al.sampler import AL_sampler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.trans import Transformer, undo_transforms
from deepchem.data import Dataset, NumpyDataset
from deepchem.utils.typing import OneOrMany
import random
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel
from sklearn.metrics import mean_squared_error
import time
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from AS_Molecule.utils.funcs import *
import math
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for AL")
    parser.add_argument("--explore", type=int, default=5)
    parser.add_argument("--exploit", type=int, default=5)
    parser.add_argument("--model_path", type=str,  default = 'AL_dft/models/model_final/')
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--unlabel_path", type=str, default = 'AL_dft/datasets/final_mxene_os/MXene_E_table.csv')
    parser.add_argument("--cif_path", type=str, default = 'AL_dft/datasets/final_mxene_os/cif_file/CONTCAR_')
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--result_path", type=str, default = 'AL_dft/result/mxene_final/')
    parser.add_argument("--transfer", type=bool)
    parser.add_argument("--x_column", type=str, default = 'Surface')
    parser.add_argument("--y_column", type=str, default = 'Eads')
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--suffix", type=str, default = '')
    args = parser.parse_args()
    return args

class QBC_sampler(AL_sampler):
  def __init__(self, args, init_ids=None, method='random', **kwargs):
    super(QBC_sampler, self).__init__(args = args, init_ids = init_ids, method = method, **kwargs)
    self.args = args
    self.core_ids = init_ids 

  def query(self, inputs, prediction = None, input_data=None, features=None, model=None):

        if self.args.explore > 0 and self.args.exploit > 0:
            new_batch_ids = self._top_query(prediction, args.exploit)
            new_batch_ids = np.concatenate((new_batch_ids, self._k_center_query(inputs, args.explore)))
        elif self.args.exploit == 0:
            new_batch_ids = self._k_center_query(inputs, args.explore)
        elif self.args.explore == 0:
            new_batch_ids = self._top_query(prediction, args.exploit)
        else:
            raise ValueError

        return new_batch_ids

  def _top_query(self, predictions, query_number):
    unlabel_pred = predictions[self.data_ids]
    unlabel_delta = abs(unlabel_pred+0.7)
    query_ids = np.argpartition(unlabel_delta, query_number)[:query_number]
    new_batch_ids = self.data_ids[query_ids]
    self.core_ids = np.sort(np.concatenate([self.core_ids, new_batch_ids]))
    self.data_ids = np.delete(self.data_ids, query_ids)
    return new_batch_ids


  def _k_center_query(self, inputs, query_number = None):
      
      if query_number or query_number==0:
        pass
      else:
        query_number = self.batch_data_num

      time0 = time.time()

      new_batch_ids_ = []
      new_batch_ids = []
      # calculate the minimum dist using a chunked way
      un_embeddings = inputs[self.data_ids]
      core_embeddings = inputs[
          self.core_ids]  # core point is the data already choosen
      min_dist = 1e5 * torch.ones(self.total_data_num).to(
          un_embeddings.device)
      min_dist[self.core_ids] = 0

      un_ebd_a = torch.sum(un_embeddings**2, dim=1)
      c_ebd_b = torch.sum(core_embeddings**2, dim=1)
      
      min_dist[self.data_ids] = un_ebd_a + torch.min(
              c_ebd_b -
              2 * un_embeddings @ core_embeddings.t(),
              dim=1)[0]

      return_dist = min_dist

      print(query_number)

      if query_number == 0:
        return return_dist
      
      
      for id in range(query_number):
          new_point_id_ = int(torch.argmax(
              min_dist[self.data_ids]))  # id relative to query_data_ids
          new_point_id = self.data_ids[new_point_id_]
          new_batch_ids_.append(new_point_id_)
          new_batch_ids.append(new_point_id)
          distance_new = torch.sum((inputs[new_point_id] - inputs)**2, dim=1)
          min_dist = torch.min(torch.stack([min_dist, distance_new], dim=0),
                                dim=0)[0]
          # print(id)

      
      self.core_ids = np.sort(np.concatenate([self.core_ids, new_batch_ids]))
      self.data_ids = np.delete(self.data_ids, new_batch_ids_)
      print('query new data {}'.format(time.time() - time0))
      return new_batch_ids


class AL_CGCNNModel(CGCNNModel):
  def __init__(self, **kwargs):
    super(AL_CGCNNModel, self).__init__(**kwargs)



  def forward_til_emb(self, dgl_graph):
    graph = dgl_graph
    # embedding node features
    node_feats = graph.ndata.pop('x')
    edge_feats = graph.edata.pop('edge_attr')
    node_feats = self.model.embedding(node_feats)

    # convolutional layer
    for conv in self.model.conv_layers:
      node_feats = conv(graph, node_feats, edge_feats)

    # pooling
    graph.ndata['updated_x'] = node_feats
    graph_feat = F.softplus(self.model.pooling(graph, 'updated_x'))
    return graph_feat

  def _emb(self, generator: Iterable[Tuple[Any, Any, Any]],
               transformers: List[Transformer], uncertainty: bool,
               other_output_types: Optional[OneOrMany[str]]):
    self._ensure_built()
    self.model.eval()
    all_emb = []
    for batch in generator:
      inputs, labels, weights = batch
      inputs, _, _ = self._prepare_batch((inputs, None, None))

      # Invoke the model.
      
      output_emb = self.forward_til_emb(inputs)
      output_emb = [t.detach().cpu() for t in output_emb]
      all_emb.extend(output_emb)
    all_emb = torch.stack(all_emb, dim = 0)
    return all_emb

  def gen_emb(self, dataset: Dataset,
               transformers: List[Transformer]= []):
    generator = self.default_generator(dataset,
                                       mode='predict',
                                       pad_batches=False)
    return self._emb(generator, [], False, ['embedding'])


def training_loop(args):
  label_df = pd.read_csv(args.label_path)
  unlabel_df = pd.read_csv(args.unlabel_path)
  label_data = label_df[args.x_column].values
  input_data = unlabel_df[args.x_column].values
  featurizer = CGCNNFeaturizer(radius = 8.0, max_neighbors=6)
  structure_list = []
  for id in input_data:
    #print(id)
    parser = CifParser(args.cif_path+id+'.cif') # <<<< Path ของ .cif
    structure = parser.get_structures()[0]
    structure_list.append(structure)  
  features = featurizer.featurize(structure_list)
  all_dataset = dc.data.DiskDataset.from_numpy(features, np.ones(len(features)))
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  init_id = list(np.nonzero(label_data[:, None] == input_data)[1])
  sampler = QBC_sampler(args = args, total_data_num = len(input_data), batch_data_num = 10, init_ids=init_id)
  nan_index = np.isnan(label_df[args.y_column].values)

  X = features[np.array(sampler.label_ids)[~nan_index]]
  y = label_df[args.y_column].values[~nan_index]
  indices = list(label_df[~nan_index].index)

  train_sample_all_class = True
  seed = args.seed
  X_train, X_valid, y_train, y_valid, ind_train, ind_valid = train_test_split(X, y, indices, train_size=0.8, random_state=seed)

  print(label_df[args.x_column][ind_train].sort_values())

  X_unlabel = features[sampler.data_ids]
  train_dataset = dc.data.DiskDataset.from_numpy(X_train, y_train)
  valid_dataset = dc.data.DiskDataset.from_numpy(X_valid, y_valid)
  unknown_ids = sampler.data_ids

  
  model = AL_CGCNNModel(args = args, mode='regression', in_edge_dim= features[0].edge_features.shape[1], batch_size=16, learning_rate=3e-4, num_conv = 6)
  best = 999
  if args.transfer:
    print('transfer successful')
    model.restore(args.model_path+str(args.iteration-1)+'/checkpoint1.pt')
  for i in range(100):
    loss = model.fit(train_dataset, nb_epoch=1)
    valid_loss = mean_squared_error(model.predict(valid_dataset), valid_dataset.y, squared=False)
    #print("Epoch %d train loss: %f" % (i, loss))
    #print("Epoch %d valid loss: %f" % (i, valid_loss))
    
    if valid_loss < best:
      best = valid_loss 
      model.save_checkpoint(model_dir=args.model_path+'/'+str(args.iteration))
  model.restore(args.model_path+str(args.iteration)+'/checkpoint1.pt')

  all_pred = np.squeeze(model.predict(all_dataset))

  if args.explore > 0 and args.exploit > 0:
      method_list = ['exploit' for i in range(args.explore)]
      method_list.extend(['explore' for i in range(args.exploit)])
  elif args.exploit == 0:
      method_list = ['explore' for i in range(args.explore)]
  elif args.explore == 0:
      method_list = ['exploit' for i in range(args.exploit)]

  all_emb = model.gen_emb(all_dataset)

  new_batch_ids = sampler.query(all_emb, all_pred, label_df, features, model)

  new_batch_df = unlabel_df.iloc[list(new_batch_ids)].drop('Unnamed: 0', axis = 1)

  new_batch_df['method'] = method_list

  std_dict = {args.x_column : input_data[unknown_ids], 'prediction' : all_pred[unknown_ids] , 'delta' :  abs(0.7+all_pred[unknown_ids])}

  std_df = pd.DataFrame(data = std_dict).sort_values(by=['delta'],  ascending=True)

  new_batch_df.to_csv(args.result_path+'new_batch_'+str(args.iteration)+args.suffix+'.csv', index=False)

  std_df.to_csv(args.result_path+'pred_dist_'+str(args.iteration)+args.suffix+'.csv')

  np.save(args.result_path+'all_emb_'+str(args.iteration)+args.suffix, all_emb.numpy())

  '''new_batch_ids, variance = sampler.query(model.gen_emb(all_dataset), all_pred, label_df, features, model)
  
  new_batch_df = unlabel_df.iloc[list(new_batch_ids)].drop('Unnamed: 0', axis = 1)

  new_batch_df['method'] = method_list

  std_dict = {args.x_column : input_data[unknown_ids], 'prediction' : all_pred[unknown_ids] , 'delta' :  abs(0.7+all_pred[unknown_ids]),'variance' : variance}

  std_df = pd.DataFrame(data = std_dict).sort_values(by=['variance'],  ascending=False)

  new_batch_df.to_csv(args.result_path+'new_batch_'+str(args.iteration)+'.csv', index=False)

  std_df.to_csv(args.result_path+'pred_std_'+str(args.iteration)+'.csv')
  '''



if __name__ == "__main__":
  args = parse_args()
  if not os.path.exists(args.model_path+'/'+str(args.iteration)):
    os.mkdir(args.model_path+'/'+str(args.iteration))
  training_loop(args)

