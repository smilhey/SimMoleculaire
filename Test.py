import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# import for unit testing
import unittest

# import functions to test
from SimMoleculaire import *

sns.set()

# unit test for K function
class TestK(unittest.TestCase):
        def test_K(self):
            trajectoires = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            self.assertEqual(K(trajectoires), 3)
            trajectoires = [[1, 2, 4], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
            self.assertEqual(K(trajectoires), 4)

# unit test for I function
class TestI(unittest.TestCase):
        def test_I(self):
            trajectoires = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            self.assertEqual(I(trajectoires, 3), [0])
            trajectoires = [[1, 2, 4], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
            self.assertEqual(I(trajectoires, 4), [0, 1])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)




