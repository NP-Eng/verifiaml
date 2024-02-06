// TODO change to ark_std?
use core::fmt::Debug;
use std::{
    ops::{Add, Div, Mul, Sub},
    vec,
};

use crate::quantization::{QLargeType, QSmallType};

pub(crate) trait InnerType: Copy {}

impl InnerType for QSmallType {}
impl InnerType for QLargeType {}
impl InnerType for u8 {}

#[derive(Debug, Clone)]
pub(crate) struct QArray<T: InnerType> {
    flattened: Vec<T>,
    shape: Vec<usize>,
    cumulative_dimensions: Vec<usize>,
}

impl<T: InnerType> QArray<T> {
    pub(crate) fn check_dimensions(&self) -> bool {
        self.flattened.len() == self.shape.iter().product::<usize>()
    }

    pub(crate) fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub(crate) fn len(&self) -> usize {
        self.flattened.len()
    }

    pub(crate) fn num_dims(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn values(&self) -> &Vec<T> {
        &self.flattened
    }

    pub(crate) fn move_values(self) -> Vec<T> {
        self.flattened
    }

    // TODO in the future, if necessary, we can remove the bound
    // <T as TryInto<S>>::Error: Debug
    // and replace unwrap() by unwrap_or(), possibly panicking or propagating
    // the error
    pub(crate) fn cast<S: InnerType>(self) -> QArray<S>
    where
        T: TryInto<S>,
        <T as TryInto<S>>::Error: Debug,
    {
        let flattened = self
            .flattened
            .into_iter()
            .map(|x| x.try_into().unwrap())
            .collect();
        QArray::new(flattened, self.shape)
    }

    // Reshapes the QArray in-place
    pub(crate) fn reshape(&mut self, shape: Vec<usize>) {
        assert_eq!(
            self.len(),
            shape.iter().product::<usize>(),
            "New shape must have the same number of elements as the original one"
        );

        self.shape = shape;
    }

    // Internal constructor that computes cumulative dimensions
    fn new(flattened: Vec<T>, shape: Vec<usize>) -> Self {
        let mut cumulative_dimensions = Vec::new();

        let mut acc = 1;

        for dim in shape.iter().rev() {
            cumulative_dimensions.push(acc);
            acc *= dim;
        }

        cumulative_dimensions.reverse();

        Self {
            flattened,
            shape,
            cumulative_dimensions,
        }
    }

    /// Takes an n-dimensional index and returns the corresponding flattened
    /// index. E.g. for a 3x3 matrix, the index (1, 2) corresponds
    /// to the flattened index 5.
    fn flatten_index(&self, index: Vec<usize>) -> usize {
        debug_assert_eq!(
            index.len(),
            self.num_dims(),
            "Index has the wrong number of dimensions"
        );

        index
            .iter()
            .zip(self.cumulative_dimensions.iter())
            .map(|(i, d)| i * d)
            .sum()
    }

    fn get(&self, index: Vec<usize>) -> T {
        self.flattened[self.flatten_index(index)]
    }
}

/************************ Operators ************************/

// Since numerical type control is essential, we implement only QArray<T> + T
// insead of the more general QArray<T> + S for any S which can be added to T,
// thus forcing the programmer to make intentional casts. The same applies to
// other operator implementations below.
impl<T: InnerType> Add<T> for QArray<T>
where
    T: Add<Output = T>,
{
    type Output = QArray<T>;

    fn add(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x + rhs).collect();
        QArray::new(flattened, self.shape)
    }
}

// Addition in the other direction cannot be implemented in the same way, cf.
// https://stackoverflow.com/questions/70220168/how-to-implement-mul-trait-for-a-custom-struct-type-to-work-in-both-ways
// There is a workaround, but it is not necessary for now
// impl<T: InnerType> ops::Add<QArray<T>> for T where T: ops::Add<Output = T>

impl<T: InnerType> Sub<T> for QArray<T>
where
    T: Sub<Output = T>,
{
    type Output = QArray<T>;

    fn sub(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x - rhs).collect();
        QArray::new(flattened, self.shape)
    }
}

impl<T: InnerType> Mul<T> for QArray<T>
where
    T: Mul<Output = T>,
{
    type Output = QArray<T>;

    fn mul(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x * rhs).collect();
        QArray::new(flattened, self.shape)
    }
}

impl<T: InnerType> Div<T> for QArray<T>
where
    T: Div<Output = T>,
{
    type Output = QArray<T>;

    fn div(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x / rhs).collect();
        QArray::new(flattened, self.shape)
    }
}

/******************* Conversion from Vec *******************/

impl<T: InnerType> From<Vec<T>> for QArray<T> {
    fn from(values: Vec<T>) -> Self {
        let l = values.len();
        QArray::new(values, vec![l])
    }
}

impl<T: InnerType> From<Vec<Vec<T>>> for QArray<T> {
    fn from(values: Vec<Vec<T>>) -> Self {
        assert!(
            values.iter().all(|x| x.len() == values[0].len()),
            "All sub-vectors must have the same length"
        );

        let shape = vec![values.len(), values[0].len()];

        let flattened = values.into_iter().flatten().collect();
        QArray::new(flattened, shape)
    }
}

// This doesn't work, cf.
// https://stackoverflow.com/questions/37347311/how-is-there-a-conflicting-implementation-of-from-when-using-a-generic-type
// impl<T: InnerType, S: InnerType> From<QArray<T>> for QArray<S> where S: From<T> {
//     fn from(input: QArray<T>) -> Self {
//         let flattened = input.move_values().into_iter().map(|x| x as S).collect();
//         Self {
//             flattened,
//             shape: input.shape,
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_index_trivial() {
        let q = QArray::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        for i in 0..9 {
            assert_eq!(q.flatten_index(vec![i]), i);
        }
    }

    #[test]
    fn test_flatten_index_2d() {
        let shape = vec![3, 3];
        let flattened: Vec<i32> = (1..=9).collect();
        let q = QArray::new(flattened, shape);
        assert_eq!(q.flatten_index(vec![0, 0]), 0);
        assert_eq!(q.flatten_index(vec![0, 1]), 1);
        assert_eq!(q.flatten_index(vec![0, 2]), 2);
        assert_eq!(q.flatten_index(vec![1, 0]), 3);
        assert_eq!(q.flatten_index(vec![1, 2]), 5);
        assert_eq!(q.flatten_index(vec![2, 0]), 6);
        assert_eq!(q.flatten_index(vec![2, 2]), 8);
    }

    #[test]
    fn test_flatten_index_3d() {
        let shape = vec![2, 3, 4];
        let flattened: Vec<i32> = (1..=24).collect();
        let q = QArray::new(flattened, shape);
        assert_eq!(q.flatten_index(vec![0, 0, 0]), 0);
        assert_eq!(q.flatten_index(vec![0, 0, 1]), 1);
        assert_eq!(q.flatten_index(vec![0, 0, 2]), 2);
        assert_eq!(q.flatten_index(vec![0, 1, 0]), 4);
        assert_eq!(q.flatten_index(vec![1, 0, 0]), 12);
        assert_eq!(q.flatten_index(vec![1, 2, 3]), 23);
    }
}
