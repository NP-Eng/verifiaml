
// TODO change to ark_std?
use std::{vec, ops::{Add, Sub, Mul, Div}};

use crate::quantization::{QSmallType, QLargeType};

pub(crate) trait InnerType: Copy {}

impl InnerType for QSmallType {}

impl InnerType for QLargeType {}

#[derive(Debug, Clone)]
pub(crate) struct QArray<T: InnerType> {
    flattened: Vec<T>,
    shape: Vec<usize>,
}

impl<T: InnerType> QArray<T> {
    
    pub(crate) fn check_dimensions(&self) -> bool {
        self.flattened.len() == self.shape.iter().product()
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

    pub(crate) fn cast<S: InnerType>(self) -> QArray<S> where S: From<T> {
        let flattened = self.flattened.into_iter().map(|x| x.into()).collect();
        QArray {
            flattened,
            shape: self.shape.clone(),
        }
    }

    // Reshapes the QArray in-place
    pub(crate) fn reshape(&mut self, shape: Vec<usize>) {
        
        assert_eq!(
            self.len(),
            shape.iter().product(),
            "New shape must have the same number of elements as the original one"
        );
        
        self.shape = shape;
    }
}

/************************ Operators ************************/

// Since numerical type control is essential, we implement only QArray<T> + T
// insead of the more general QArray<T> + S for any S which can be added to T,
// thus forcing the programmer to make intentional casts. The same applies to
// other operator implementations below.
impl<T: InnerType> Add<T> for QArray<T> where T: Add<Output = T>{
    type Output = QArray<T>;

    fn add(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x + rhs).collect();
        QArray {
            flattened,
            shape: self.shape,
        }
    }
}

// Addition in the other direction cannot be implemented in the same way, cf.
// https://stackoverflow.com/questions/70220168/how-to-implement-mul-trait-for-a-custom-struct-type-to-work-in-both-ways
// There is a workaround, but it is not necessary for now
// impl<T: InnerType> ops::Add<QArray<T>> for T where T: ops::Add<Output = T>

impl<T: InnerType> Sub<T> for QArray<T> where T: Sub<Output = T>{
    type Output = QArray<T>;

    fn sub(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x - rhs).collect();
        QArray {
            flattened,
            shape: self.shape,
        }
    }
}

impl<T: InnerType> Mul<T> for QArray<T> where T: Mul<Output = T>{
    type Output = QArray<T>;

    fn mul(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x * rhs).collect();
        QArray {
            flattened,
            shape: self.shape,
        }
    }
}

impl<T: InnerType> Div<T> for QArray<T> where T: Div<Output = T>{
    type Output = QArray<T>;

    fn div(self, rhs: T) -> QArray<T> {
        let flattened = self.flattened.into_iter().map(|x| x / rhs).collect();
        QArray {
            flattened,
            shape: self.shape,
        }
    }
}

/******************* Conversion from Vec *******************/

impl<T: InnerType> From<Vec<T>> for QArray<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            shape: vec![values.len()],
            flattened: values,
        }
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
        Self { flattened, shape }
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
