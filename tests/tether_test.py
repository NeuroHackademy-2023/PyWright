import unittest
import PyWright.tether as tether
import numpy as np
import pandas as pd
import nibabel as nib
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_group_consensus_mats_output_shape(self):
        func_mats = np.load('test_data/func_mats.npy')
        sc_mats = np.load('test_data/sc_mats.npy')
        atlas_hemiid = pd.from_csv('test_data/atlas_meta.csv')
        dist_mat = np.load('test_data/dist_mat.npy')
        fmat, scmat = tether.group_consensus_mats(func_mats, sc_mats, atlas_hemiid, dist_mat)
        self.assertEqual(fmat.shape, scmat.shape)
        self.assertEqual(fmat.shape, (len(func_mats[0]), len(func_mats[0])))
        self.assertEqual(scmat.shape, (len(sc_mats[0]), len(sc_mats[0])))

    def test_group_consensus_mats_output_symetrical(self):
        func_mats = np.load('test_data/func_mats.npy')
        sc_mats = np.load('test_data/sc_mats.npy')
        atlas_hemiid = pd.from_csv('test_data/atlas_meta.csv')
        dist_mat = np.load('test_data/dist_mat.npy')
        fmat, scmat = tether.group_consensus_mats(func_mats, sc_mats, atlas_hemiid, dist_mat)
        self.assertTrue(np.allclose(fmat, fmat.T))
        self.assertTrue(np.allclose(scmat, scmat.T))

    def test_group_consensus_mats_output_sc_binary(self):
        func_mats = np.load('test_data/func_mats.npy')
        sc_mats = np.load('test_data/sc_mats.npy')
        atlas_hemiid = pd.from_csv('test_data/atlas_meta.csv')
        dist_mat = np.load('test_data/dist_mat.npy')
        fmat, scmat = tether.group_consensus_mats(func_mats, sc_mats, atlas_hemiid, dist_mat)
        self.assertTrue(np.allclose(scmat, scmat.astype(bool)))
    def test_get_predictor_vectors_output_shape(self):
        mats = np.load('test_data/sc_mats.npy')
        nodes = 10
        vectors = tether.get_predictor_vectors(mats, nodes)
        self.assertEqual(len(vectors), len(mats))
        self.assertEqual(vectors[0].shape, (len(mats[0]), 1))

    def test_euclidean_distance_output_shape(self):
        parcellation = 'test_data/parcellation.nii.gz'
        lables = np.unique(nib.load(parcellation).get_data())
        dist_mat = tether.euclidean_distance(parcellation)
        self.assertEqual(dist_mat.shape, (len(lables), len(lables)))

    def test_euclidean_distance_output_range(self):
        parcellation = 'test_data/parcellation.nii.gz'
        parc_shape = np.shape(nib.load(parcellation).get_data())
        dist_mat = tether.euclidean_distance(parcellation)
        self.assertGreaterEqual(dist_mat.min(), 0)
        self.assertLessEqual(dist_mat.max(), np.sqrt(parc_shape[0]**2 + parc_shape[1]**2 + parc_shape[2]**2))

    def test_euclidean_distance_output_symmetry(self):
        parcellation = 'test_data/parcellation.nii.gz'
        dist_mat = tether.euclidean_distance(parcellation)
        self.assertTrue(np.allclose(dist_mat, dist_mat.T))

    def test_euclidean_distance_output_diagonal(self):
        parcellation = 'test_data/parcellation.nii.gz'
        dist_mat = tether.euclidean_distance(parcellation)
        self.assertTrue(np.allclose(dist_mat.diagonal(), 0))

    def test_euclidean_distance_output_zeros(self): # test for zeros in the matrix except for the diagonal
        parcellation = 'test_data/parcellation.nii.gz'
        dist_mat = tether.euclidean_distance(parcellation)
        self.assertFalse(np.any(np.triu(dist_mat, 1) == 0))

    def test_communicability_output_shape(self):
        sc_mat = np.load('test_data/sc_mat.npy')
        comm_mat = tether.communicability(sc_mat)
        self.assertEqual(comm_mat.shape, sc_mat.shape)

    def test_communicability_output_symmetry(self):
        sc_mat = np.load('test_data/sc_mat.npy')
        comm_mat = tether.communicability(sc_mat)
        self.assertTrue(np.allclose(comm_mat, comm_mat.T))

    def test_shortest_path_output_shape(self):
        sc_mat = np.load('test_data/sc_mat.npy')
        path_mat = tether.shortest_path(sc_mat)
        self.assertEqual(path_mat.shape, sc_mat.shape)

    def test_shortest_path_output_symmetry(self):
        sc_mat = np.load('test_data/sc_mat.npy')
        path_mat = tether.shortest_path(sc_mat)
        self.assertTrue(np.allclose(path_mat, path_mat.T))

if __name__ == '__main__':
    unittest.main()
