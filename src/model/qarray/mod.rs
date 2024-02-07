use ark_std::fmt::Debug;
use ark_std::ops::{Add, Div, Mul, Sub};
use ark_std::vec;
use ark_std::vec::Vec;

use crate::quantization::{QLargeType, QSmallType};

#[cfg(test)]
mod tests;

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

    // TODO it's possible this can be handled more elegantly with Deref, like
    // the reference codebase does
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
    pub(crate) fn new(flattened: Vec<T>, shape: Vec<usize>) -> Self {
        assert!(shape.len() > 0, "Arrays cannot be zero-dimensional");

        let mut cumulative_dimensions = Vec::with_capacity(shape.len());

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

    pub(crate) fn get(&self, index: Vec<usize>) -> T {
        self.flattened[self.flatten_index(index)]
    }

    /// For each dimension of self.shape, either pad the QArray with `value`
    /// (if the new size is larger than the original one) or truncate it (if
    /// the new size is smaller than or equal to the original one).
    pub(crate) fn compact_resize(&self, new_shape: Vec<usize>, value: T) -> QArray<T> {
        let old_shape = &self.shape;

        assert_eq!(
            new_shape.len(),
            old_shape.len(),
            "New and old shape must have the same number of dimensions, but they have {} and {}, respectively",
            new_shape.len(),
            old_shape.len(),
        );

        // compute cumulative dimensions of the qarray
        let mut new_cumulative_dimensions = Vec::with_capacity(new_shape.len());
        let mut acc = 1;

        for dim in new_shape.iter().rev() {
            new_cumulative_dimensions.push(acc);
            acc *= dim;
        }

        new_cumulative_dimensions.reverse();

        let flattened = compact_resize_internal(
            &self.flattened,
            &old_shape,
            &new_shape,
            &self.cumulative_dimensions,
            &new_cumulative_dimensions,
            new_shape[new_shape.len() - 1],
            value,
        );

        QArray::new(flattened, new_shape)
    }
}

/************************* Padding *************************/
// TODO this can perhaps be done more efficiently, e.g. by performing all data
// manipulation in-place using indices rather than creating new vectors
fn compact_resize_internal<T: InnerType>(
    data: &Vec<T>,
    old_shape: &[usize],
    new_shape: &[usize],
    old_cumulative_dimensions: &[usize],
    new_cumulative_dimensions: &[usize],
    final_new_size: usize,
    value: T,
) -> Vec<T> {
    // Base case at length 1 (rather than 0) is slightly less elegant but more
    // efficient
    if new_shape.len() == 1 {
        let mut new_data = data.clone();
        new_data.resize(final_new_size, value);
        return new_data;
    }

    // TODO better modify in-place with chunks_exact_mut?
    let subarrays = data.chunks_exact(old_cumulative_dimensions[0]);

    let padded: Vec<T> = if new_shape[0] <= old_shape[0] {
        subarrays
            .take(new_shape[0])
            .flat_map(|subarray| {
                compact_resize_internal(
                    &subarray.to_vec(),
                    &old_shape[1..],
                    &new_shape[1..],
                    &old_cumulative_dimensions[1..],
                    &new_cumulative_dimensions[1..],
                    final_new_size,
                    value,
                )
            })
            .collect()
    } else {
        subarrays
            .flat_map(|subarray| {
                compact_resize_internal(
                    &subarray.to_vec(),
                    &old_shape[1..],
                    &new_shape[1..],
                    &old_cumulative_dimensions[1..],
                    &new_cumulative_dimensions[1..],
                    final_new_size,
                    value,
                )
            })
            .chain(
                vec![value; (new_shape[0] - old_shape[0]) * new_cumulative_dimensions[0]]
                    .into_iter(),
            )
            .collect()
    };

    padded
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
