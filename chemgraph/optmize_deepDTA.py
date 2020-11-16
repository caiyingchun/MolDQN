# GLH

"""Optimize molecule with deepDTA as reward function. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
from absl import app
from absl import flags
from rdkit import Chem

from chemgraph.dqn import deep_q_networks
from chemgraph.dqn import molecules as molecules_mdp
from chemgraph.dqn import run_dqn
from chemgraph.dqn.py import molecules
from chemgraph.dqn.tensorflow_core import core

import sys
#print("path before insert: ", sys.path)
sys.path.insert(1, '/home/c304004/PycharmProjects/deepDTA-mod/model_application/')
#print("path after insert: ", sys.path)
from loadModelTest import *

flags.DEFINE_float('gamma', 0.999, 'discount')
FLAGS = flags.FLAGS


class Molecule(molecules_mdp.Molecule):
  """deepDTA reward Molecule."""

  def _reward(self):
    # SMILES MOLECULE : molecule = self._state... Need this to give to deepDTA
    # Mol obj: molecule = Chem.MolFromSmiles(self._state)... Need this to run mol_dqn?
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return 0.0
    return

    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    preds = model_application.loadModelTest.makePredictions(FLAGS)
    return preds


def main(argv):
  del argv
  if FLAGS.hparams is not None:
    with open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  environment = Molecule(
      atom_types=set(hparams.atom_types),
      init_mol=FLAGS.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode)

  dqn = deep_q_networks.DeepQNetwork(
      input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
      q_fn=functools.partial(
          deep_q_networks.multi_layer_model, hparams=hparams),
      optimizer=hparams.optimizer,
      grad_clipping=hparams.grad_clipping,
      num_bootstrap_heads=hparams.num_bootstrap_heads,
      gamma=hparams.gamma,
      epsilon=1.0)

  run_dqn.run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn,
  )

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
