"""
Tests for msmbuilder.metrics.fast_cdist and msmbuilder.metrics.fast_pdist
"""

import numpy as np

from scipy.spatial.distance import pdist, cdist
from msmbuilder.metrics.core import fast_pdist, fast_cdist
from msmbuilder.testing import *
import time

class test_pdist():
    def setUp(self):
        np.random.seed(42)
        self.Xd = np.random.randn(1000, 10)
        self.Xb = np.asarray(np.random.randint(2, size=(1000, 10)), dtype=np.bool)
    
    def test_seuclidean_1(self):
        reference = pdist(self.Xd, metric='seuclidean')
        testing = fast_pdist(self.Xd, metric='seuclidean')
        eq(reference, testing)
    
    def test_seuclidean_2(self):
        V = np.random.randn(self.Xd.shape[1])
        reference = pdist(self.Xd, metric='seuclidean', V=V)
        testing = fast_pdist(self.Xd, metric='seuclidean', V=V)
        eq(reference, testing)
        
    def test_mahalanobis_1(self):
        reference = pdist(self.Xd, metric='mahalanobis')
        testing = fast_pdist(self.Xd, metric='mahalanobis')
        eq(reference, testing)
    
    def test_mahalanobis_2(self):
        VI = np.random.randn(self.Xd.shape[0], self.Xd.shape[0])
        reference = pdist(self.Xd, metric='mahalanobis', VI=VI)
        testing = fast_pdist(self.Xd, metric='mahalanobis', VI=VI)
        eq(reference, testing)
    
    def test_minkowski(self):
        p = np.random.randint(10)
        reference = pdist(self.Xd, metric='minkowski', p=p)
        testing = fast_pdist(self.Xd, metric='minkowski', p=p)
        eq(reference, testing)
    
    def test_cosine(self):
        reference = pdist(self.Xd, metric='cosine')
        testing = fast_pdist(self.Xd, metric='cosine')
        eq(reference, testing)
    
    def test_cityblock(self):
        reference = pdist(self.Xd, metric='cityblock')
        testing = fast_pdist(self.Xd, metric='cityblock')
        eq(reference, testing)
    
    def test_correlation(self):
        reference = pdist(self.Xd, metric='correlation')
        testing = fast_pdist(self.Xd, metric='correlation')
        eq(reference, testing)
    
    def test_euclidean(self):
        reference = pdist(self.Xd, metric='euclidean')
        testing = fast_pdist(self.Xd, metric='euclidean')
        eq(reference, testing)
    
    def test_sqeuclidean(self):
        reference = pdist(self.Xd, metric='sqeuclidean')
        testing = fast_pdist(self.Xd, metric='sqeuclidean')
        eq(reference, testing)
    
    def test_chebychev(self):
        reference = pdist(self.Xd, metric='chebychev')
        testing = fast_pdist(self.Xd, metric='chebychev')
        eq(reference, testing)
    
    def test_hamming_1(self):
        reference = pdist(self.Xd, metric='hamming')
        testing = fast_pdist(self.Xd, metric='hamming')
        eq(reference, testing)
    
    def test_hamming_2(self):
        reference = pdist(self.Xb, metric='hamming')
        testing = fast_pdist(self.Xb, metric='hamming')
        eq(reference, testing)
    
    def test_jaccard_1(self):
        reference = pdist(self.Xd, metric='jaccard')
        testing = fast_pdist(self.Xd, metric='jaccard')
        eq(reference, testing)
    
    def test_jaccard_2(self):
        reference = pdist(self.Xb, metric='jaccard')
        testing = fast_pdist(self.Xb, metric='jaccard')
        eq(reference, testing)
    
    @expected_failure
    def test_canberra(self):
        reference = pdist(self.Xd, metric='canberra')
        testing = fast_pdist(self.Xd, metric='canberra')
        eq(reference, testing)
    
    def test_braycurtis(self):
        reference = pdist(self.Xd, metric='braycurtis')
        testing = fast_pdist(self.Xd, metric='braycurtis')
        eq(reference, testing)
    
    def test_yule(self):
        reference = pdist(self.Xb, metric='yule')
        testing = fast_pdist(self.Xb, metric='yule')
        eq(reference, testing)
    
    def test_matching(self):
        reference = pdist(self.Xb, metric='matching')
        testing = fast_pdist(self.Xb, metric='matching')
        eq(reference, testing)
        
    def test_kulsinski(self):
        reference = pdist(self.Xb, metric='kulsinski')
        testing = fast_pdist(self.Xb, metric='kulsinski')
        eq(reference, testing)
                
    def test_dice(self):
        reference = pdist(self.Xb, metric='dice')
        testing = fast_pdist(self.Xb, metric='dice')
        eq(reference, testing)
                    
    def test_rogerstanimoto(self):
        reference = pdist(self.Xb, metric='rogerstanimoto')
        testing = fast_pdist(self.Xb, metric='rogerstanimoto')
        eq(reference, testing)
                        
    def test_russellrao(self):
        reference = pdist(self.Xb, metric='russellrao')
        testing = fast_pdist(self.Xb, metric='russellrao')
        eq(reference, testing)
                            
    def test_sokalmichener(self):
        reference = pdist(self.Xb, metric='sokalmichener')
        testing = fast_pdist(self.Xb, metric='sokalmichener')
        eq(reference, testing)
                                
    def test_sokalsneath(self):
        reference = pdist(self.Xb, metric='sokalsneath')
        testing = fast_pdist(self.Xb, metric='sokalsneath')
        eq(reference, testing)


class test_cdist():
    def setUp(self):
        np.random.seed(42)
        self.Xd1 = np.random.randn(1000, 10)
        self.Xd2 = np.random.randn(1, 10)
        self.Xb1 = np.asarray(np.random.randint(2, size=(1000, 10)), dtype=np.bool)
        self.Xb2 = np.asarray(np.random.randint(2, size=(1, 10)), dtype=np.bool)
    
    def test_seuclidean_2(self):
        V = np.random.randn(self.Xd1.shape[1])
        reference = cdist(self.Xd1, self.Xd2, metric='seuclidean', V=V)
        testing = fast_cdist(self.Xd1, self.Xd2, metric='seuclidean', V=V)
        eq(reference, testing)
    
    def test_mahalanobis_2(self):
        VI = np.random.randn(self.Xd1.shape[0], self.Xd1.shape[0])
        reference = cdist(self.Xd1, self.Xd2, metric='mahalanobis', VI=VI)
        testing = fast_cdist(self.Xd1, self.Xd2, metric='mahalanobis', VI=VI)
        eq(reference, testing)
    
    def test_minkowski(self):
        p = np.random.randint(10)
        reference = cdist(self.Xd1, self.Xd2, metric='minkowski', p=p)
        testing = fast_cdist(self.Xd1, self.Xd2, metric='minkowski', p=p)
        eq(reference, testing)
    
    def test_cosine(self):
        reference = cdist(self.Xd1, self.Xd2, metric='cosine')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='cosine')
        eq(reference, testing)
    
    def test_cityblock(self):
        reference = cdist(self.Xd1, self.Xd2, metric='cityblock')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='cityblock')
        eq(reference, testing)
    
    def test_correlation(self):
        reference = cdist(self.Xd1, self.Xd2, metric='correlation')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='correlation')
        eq(reference, testing)
    
    def test_euclidean(self):
        reference = cdist(self.Xd1, self.Xd2, metric='euclidean')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='euclidean')
        eq(reference, testing)
    
    def test_sqeuclidean(self):
        reference = cdist(self.Xd1, self.Xd2, metric='sqeuclidean')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='sqeuclidean')
        eq(reference, testing)
    
    def test_chebychev(self):
        reference = cdist(self.Xd1, self.Xd2, metric='chebychev')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='chebychev')
        eq(reference, testing)
    
    def test_hamming_1(self):
        reference = cdist(self.Xd1, self.Xd2, metric='hamming')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='hamming')
        eq(reference, testing)
    
    def test_hamming_2(self):
        reference = cdist(self.Xb1, self.Xb2, metric='hamming')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='hamming')
        eq(reference, testing)
    
    @skip('jaccard not implemented in fast_cdist')
    def test_jaccard_1(self):
        reference = cdist(self.Xd1, self.Xd2, metric='jaccard')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='jaccard')
        eq(reference, testing)
    
    @skip('jaccard not implemented in fast_cdist')
    def test_jaccard_2(self):
        reference = cdist(self.Xb1, self.Xb2, metric='jaccard')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='jaccard')
        eq(reference, testing)
    
    @skip('canberra is not working')
    def test_canberra(self):
        reference = cdist(self.Xd1, self.Xd2, metric='canberra')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='canberra')
        eq(reference, testing)
    
    def test_braycurtis(self):
        reference = cdist(self.Xd1, self.Xd2, metric='braycurtis')
        testing = fast_cdist(self.Xd1, self.Xd2, metric='braycurtis')
        eq(reference, testing)
    
    def test_yule(self):
        reference = cdist(self.Xb1, self.Xb2, metric='yule')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='yule')
        eq(reference, testing)
    
    def test_matching(self):
        reference = cdist(self.Xb1, self.Xb2, metric='matching')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='matching')
        eq(reference, testing)
    
    @skip('kulsinski not implemented in fast_cdist')
    def test_kulsinski(self):
        reference = cdist(self.Xb1, self.Xb2, metric='kulsinski')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='kulsinski')
        eq(reference, testing)
    
    def test_dice(self):
        reference = cdist(self.Xb1, self.Xb2, metric='dice')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='dice')
        eq(reference, testing)
    
    def test_rogerstanimoto(self):
        reference = cdist(self.Xb1, self.Xb2, metric='rogerstanimoto')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='rogerstanimoto')
        eq(reference, testing)
    
    def test_russellrao(self):
        reference = cdist(self.Xb1, self.Xb2, metric='russellrao')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='russellrao')
        eq(reference, testing)
    
    def test_sokalmichener(self):
        reference = cdist(self.Xb1, self.Xb2, metric='sokalmichener')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='sokalmichener')
        eq(reference, testing)
    
    def test_sokalsneath(self):
        reference = cdist(self.Xb1, self.Xb2, metric='sokalsneath')
        testing = fast_cdist(self.Xb1, self.Xb2, metric='sokalsneath')
        eq(reference, testing)
