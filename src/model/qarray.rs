
use std::vec;

use crate::quantization::QSmallType;

pub(crate) struct QArray(Vec<Vec<Vec<QSmallType>>>);

impl QArray {
    // TODO change error type
    /// Check that all sub– and subsubarrays have the correct length
    pub fn check_dimensions(&self) -> Result<Vec<usize>, String> {
        let (m, n, l) = self.raw_dimensions();

        for i in 0..m {
            if self.0[i].len() != n {
                return Err(format!(
                    "Subarray with index {} has length {}, but {} was expected",
                    i,
                    self.0[i].len(),
                    n
                ));
            }

            for j in 0..n {
                if self.0[i][j].len() != l {
                    return Err(format!(
                        "Subsubarray with index ({}, {}) has length {}, but {} was expected",
                        i,
                        j,
                        self.0[i][j].len(),
                        l
                    ));
                }
            }
        }

        Ok(self.dimensions())
    }

    /// Return array dimensions without checking sub- or subsubarray lengths
    pub(crate) fn dimensions(&self) -> Vec<usize> {
        let (m, n, l) = self.raw_dimensions();

        if m == 1 {
            if n == 1 {
                vec![l]
            } else {
                vec![n, l]
            }
        } else {
            vec![m, n, l]
        }
    }

    fn raw_dimensions(&self) -> (usize, usize, usize) {
        (self.0.len(), self.0[0].len(), self.0[0][0].len())
    }

    pub(crate) fn values(&self) -> &Vec<Vec<Vec<QSmallType>>> {
        &self.0
    }

    pub(crate) fn move_values(self) -> Vec<Vec<Vec<QSmallType>>> {
        self.0
    }
}

impl From<Vec<QSmallType>> for QArray {
    fn from(value: Vec<QSmallType>) -> Self {
        Self(vec![vec![value]])
    }
}

impl From<Vec<Vec<QSmallType>>> for QArray {
    fn from(value: Vec<Vec<QSmallType>>) -> Self {
        Self(vec![value])
    }
}

impl From<Vec<Vec<Vec<QSmallType>>>> for QArray {
    fn from(value: Vec<Vec<Vec<QSmallType>>>) -> Self {
        Self(value)
    }
}
