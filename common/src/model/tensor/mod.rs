use std::ops::{AddAssign, BitAnd, DivAssign, MulAssign, Shl, Shr, SubAssign};

use ark_std::any::type_name;
use ark_std::cmp::PartialOrd;
use ark_std::fmt;
use ark_std::fmt::Debug;
use ark_std::mem;
use ark_std::ops::Index;
use ark_std::ops::{Add, Div, Mul, Sub};
use ark_std::vec;
use ark_std::vec::Vec;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json;

use crate::quantization::QScaleType;

#[cfg(test)]
mod tests;

const TENSOR_NESTED_TAB: &str = "    ";

pub trait Integral:
    Copy
    + Serialize
    + DeserializeOwned
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + BitAnd<Output = Self>
    + Into<Self::Double>
{
    // We can't simply require Double: Integral, as that would create an
    // infinite chain
    type Double: Copy
        + Debug
        + TryInto<Self>
        + Mul<Output = Self::Double>
        + Div<Output = Self::Double>
        + Add<Output = Self::Double>
        + Sub<Output = Self::Double>
        + Shl<usize, Output = Self::Double>
        + Shr<usize, Output = Self::Double>;

    const ZERO: Self;
    const ONE: Self;
    const ONE_DOUBLE: Self::Double;
    const MIN: Self;
    const MAX: Self;
    const BITS: usize;

    // TODO this should be removed once  floating requantisation is made generic
    fn from_qscaletype(x: QScaleType) -> Self;
    fn to_qscaletype(&self) -> QScaleType;
}

#[macro_export]
macro_rules! impl_integral {
    ( $t1:ty, $t2:ty ) => {
        impl Integral for $t1 {
            type Double = $t2;

            const ZERO: Self = 0;
            const ONE: Self = 1;
            const ONE_DOUBLE: Self::Double = 1;
            const MIN: Self = Self::MIN;
            const MAX: Self = Self::MAX;
            const BITS: usize = 8 * mem::size_of::<Self>();

            fn from_qscaletype(x: QScaleType) -> Self {
                x as Self
            }

            fn to_qscaletype(&self) -> QScaleType {
                *self as QScaleType
            }
        }
    };
}

impl_integral!(i8, i16);
impl_integral!(i32, i64);

pub trait SmallNIO: Integral + Into<Self::LT> {
    type LT: Integral + TryInto<Self>;
}

impl SmallNIO for i8 {
    type LT = i32;
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Tensor<T> {
    #[serde(rename = "f")]
    flattened: Vec<T>,
    #[serde(rename = "s")]
    shape: Vec<usize>,
    #[serde(rename = "c")]
    cumulative_dimensions: Vec<usize>,
}

#[derive(Clone)]
pub enum NIOTensor<ST: SmallNIO> {
    S(Tensor<ST>),
    L(Tensor<ST::LT>),
}

// indexing syntax tensor[idx] for Tensor
impl<T: Integral> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.flattened[index]
    }
}

impl<T> Tensor<T> {
    pub fn check_dimensions(&self) -> bool {
        self.flattened.len() == self.shape.iter().product::<usize>()
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.flattened.len()
    }

    pub fn num_dims(&self) -> usize {
        self.shape.len()
    }

    pub fn values(&self) -> &Vec<T> {
        &self.flattened
    }

    // TODO it's possible this can be handled more elegantly with Deref, like
    // the reference codebase does
    pub fn move_values(self) -> Vec<T> {
        self.flattened
    }

    // Reshapes the Tensor in-place
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        assert_eq!(
            self.len(),
            new_shape.iter().product::<usize>(),
            "New shape must have the same number of elements as the original one"
        );

        // Recomputing cumulative dimensions
        let mut cumulative_dimensions = Vec::with_capacity(new_shape.len());

        let mut acc = 1;

        for dim in new_shape.iter().rev() {
            cumulative_dimensions.push(acc);
            acc *= dim;
        }

        cumulative_dimensions.reverse();
        self.cumulative_dimensions = cumulative_dimensions;

        // Setting the new shape itself
        self.shape = new_shape;
    }

    // Internal constructor that computes cumulative dimensions
    pub fn new(flattened: Vec<T>, shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Arrays cannot be zero-dimensional");
        assert_eq!(
            flattened.len(),
            shape.iter().product::<usize>(),
            "Incorrect shape {:?} for data of length {}",
            shape,
            flattened.len()
        );

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
    #[allow(dead_code)]
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
}

/********************** Serialization **********************/
impl<T: Serialize + DeserializeOwned> Tensor<T> {
    pub fn write(&self, path: &str) {
        let mut writer = std::fs::File::create(path).unwrap();
        serde_json::to_writer(&mut writer, self).unwrap();
    }

    pub fn read(path: &str) -> Tensor<T> {
        let reader = std::fs::File::open(path).unwrap();
        serde_json::from_reader(reader).unwrap()
    }

    pub fn write_multiple(tensors: &[&Tensor<T>], paths: &[&str]) {
        for (tensor, path) in tensors.iter().zip(paths.iter()) {
            tensor.write(path);
        }
    }

    pub fn read_multiple(paths: &[&str]) -> Vec<Tensor<T>> {
        paths.iter().map(|path| Tensor::read(path)).collect()
    }

    pub fn write_list(tensors: &[&Tensor<T>], path: &str) {
        let mut writer = std::fs::File::create(path).unwrap();
        serde_json::to_writer(&mut writer, tensors).unwrap();
    }

    pub fn read_list(path: &str) -> Vec<Tensor<T>> {
        let reader = std::fs::File::open(path).unwrap();
        serde_json::from_reader(reader).unwrap()
    }
}

impl<T: Copy> Tensor<T> {
    /// For each dimension of self.shape, either pad the Tensor with `value`
    /// (if the new size is larger than the original one) or truncate it (if
    /// the new size is smaller than or equal to the original one).
    pub fn compact_resize(&self, new_shape: Vec<usize>, value: T) -> Tensor<T> {
        let old_shape = &self.shape;

        assert_eq!(
            new_shape.len(),
            old_shape.len(),
            "New and old shape must have the same number of dimensions, but they have {} and {}, respectively",
            new_shape.len(),
            old_shape.len(),
        );

        // compute cumulative dimensions of the tensor
        let mut new_cumulative_dimensions = Vec::with_capacity(new_shape.len());
        let mut acc = 1;

        for dim in new_shape.iter().rev() {
            new_cumulative_dimensions.push(acc);
            acc *= dim;
        }

        new_cumulative_dimensions.reverse();

        let flattened = compact_resize_internal(
            &self.flattened,
            old_shape,
            &new_shape,
            &self.cumulative_dimensions,
            &new_cumulative_dimensions,
            new_shape[new_shape.len() - 1],
            value,
        );

        Tensor::new(flattened, new_shape)
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, index: Vec<usize>) -> T {
        self.flattened[self.flatten_index(index)]
    }

    pub fn cast<S>(&self) -> Tensor<S>
    where
        T: TryInto<S>,
        <T as TryInto<S>>::Error: Debug,
    {
        let flattened = self
            .flattened
            .iter()
            .map(|x| TryInto::<S>::try_into(*x).unwrap())
            .collect();
        Tensor::new(flattened, self.shape.clone())
    }
}

/************************* Padding *************************/
// TODO this can perhaps be done more efficiently, e.g. by performing all data
// manipulation in-place using indices rather than creating new vectors
fn compact_resize_internal<T: Copy>(
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
        let mut recursed: Vec<T> = subarrays
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
            .collect();

        recursed.resize(
            recursed.len() + (new_shape[0] - old_shape[0]) * new_cumulative_dimensions[0],
            value,
        );

        recursed
    };

    padded
}

/************************ Operators ************************/
// Since numerical type control is essential, we implement only Tensor<T> + T
// insead of the more general Tensor<T> + S for any S which can be added to T,
// thus forcing the programmer to make intentional casts. The same applies to
// other operator implementations below.
impl<T: Integral> Add<T> for Tensor<T>
where
    T: Add<Output = T>,
{
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Tensor<T> {
        let flattened = self.flattened.into_iter().map(|x| x + rhs).collect();
        Tensor::new(flattened, self.shape)
    }
}

// Addition in the other direction cannot be implemented in the same way, cf.
// https://stackoverflow.com/questions/70220168/how-to-implement-mul-trait-for-a-custom-struct-type-to-work-in-both-ways
// There is a workaround, but it is not necessary for now
// impl<T: InnerType> ops::Add<Tensor<T>> for T where T: ops::Add<Output = T>

impl<T: Integral> Sub<T> for Tensor<T>
where
    T: Sub<Output = T>,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Tensor<T> {
        let flattened = self.flattened.into_iter().map(|x| x - rhs).collect();
        Tensor::new(flattened, self.shape)
    }
}

impl<T: Integral> Mul<T> for Tensor<T>
where
    T: Mul<Output = T>,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Tensor<T> {
        let flattened = self.flattened.into_iter().map(|x| x * rhs).collect();
        Tensor::new(flattened, self.shape)
    }
}

impl<T: Integral> Div<T> for Tensor<T>
where
    T: Div<Output = T>,
{
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Tensor<T> {
        let flattened = self.flattened.into_iter().map(|x| x / rhs).collect();
        Tensor::new(flattened, self.shape)
    }
}

/******************* Conversion from Vec *******************/
impl<T> From<Vec<T>> for Tensor<T> {
    fn from(values: Vec<T>) -> Self {
        let l = values.len();
        Tensor::new(values, vec![l])
    }
}

impl<T> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(values: Vec<Vec<T>>) -> Self {
        assert!(
            values.iter().all(|x| x.len() == values[0].len()),
            "All sub-vectors must have the same length"
        );

        let shape = vec![values.len(), values[0].len()];

        let flattened = values.into_iter().flatten().collect();
        Tensor::new(flattened, shape)
    }
}

/************************* Display *************************/
impl<T: Integral> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor ({}). Shape: {:?}. Data:",
            type_name::<T>(),
            self.shape
        )?;

        if self.shape.len() == 1 {
            return write!(f, " {:?}", self.flattened);
        }

        writeln!(f)?;

        print_flat_data(
            f,
            &self.flattened,
            &self.cumulative_dimensions,
            self.cumulative_dimensions.len(),
            self.cumulative_dimensions.len(),
        )
    }
}

fn print_flat_data<T: Integral>(
    f: &mut fmt::Formatter,
    data: &[T],
    cumulative_dimensions: &[usize],
    len: usize,
    original_len: usize,
) -> fmt::Result {
    if len == 0 {
        return Ok(());
    }

    // Base case
    if len == 1 {
        return writeln!(
            f,
            "{}{:?}",
            TENSOR_NESTED_TAB.repeat(original_len - 1),
            data
        );
    }

    if len != original_len {
        writeln!(f, "{}[", TENSOR_NESTED_TAB.repeat(original_len - len))?;
    }

    let subarrays = data.chunks_exact(cumulative_dimensions[0]);

    for subarray in subarrays {
        print_flat_data(
            f,
            subarray,
            &cumulative_dimensions[1..],
            len - 1,
            original_len,
        )?;
    }

    if len != original_len {
        writeln!(f, "{}]", TENSOR_NESTED_TAB.repeat(original_len - len))?;
    }

    Ok(())
}

/*********************** Comparisons ***********************/
// We follow the convention (e.g. in numpy) that `maximum` and `minimum`
// compare an array to a single element (element-wise); whereas `max` and `min`
// (not implemented) compare two equally sized arrays element-wise.
impl<T: Integral + PartialOrd> Tensor<T> {
    pub fn maximum(&self, x: T) -> Tensor<T> {
        let flattened_max: Vec<T> = self
            .flattened
            .iter()
            .map(|y| if *y >= x { *y } else { x })
            .collect();

        // Construct the new Tensor directly to avoid recomputation of
        // cumulative dimensions
        Tensor {
            flattened: flattened_max,
            shape: self.shape.clone(),
            cumulative_dimensions: self.cumulative_dimensions.clone(),
        }
    }

    pub fn minimum(&self, x: T) -> Tensor<T> {
        let flattened_min: Vec<T> = self
            .flattened
            .iter()
            .map(|y| if *y <= x { *y } else { x })
            .collect();

        // Construct the new Tensor directly to avoid recomputation of
        // cumulative dimensions
        Tensor {
            flattened: flattened_min,
            shape: self.shape.clone(),
            cumulative_dimensions: self.cumulative_dimensions.clone(),
        }
    }
}

/************************ NIOTensor ***********************/
impl<ST: SmallNIO> NIOTensor<ST> {
    #[inline]
    pub fn unwrap_small(self) -> Tensor<ST> {
        match self {
            NIOTensor::S(s) => s,
            _ => panic!("Expected S variant"),
        }
    }

    #[inline]
    pub fn unwrap_large(self) -> Tensor<ST::LT> {
        match self {
            NIOTensor::L(l) => l,
            _ => panic!("Expected L variant"),
        }
    }

    #[inline]
    pub fn ref_small(&self) -> &Tensor<ST> {
        match self {
            NIOTensor::S(s) => s,
            _ => panic!("Expected S variant"),
        }
    }

    #[inline]
    pub fn ref_large(&self) -> &Tensor<ST::LT> {
        match self {
            NIOTensor::L(l) => l,
            _ => panic!("Expected L variant"),
        }
    }

    #[inline]
    pub fn variant_name(&self) -> &'static str {
        match self {
            NIOTensor::S(_) => "NIOTensor::S",
            NIOTensor::L(_) => "NIOTensor::L",
        }
    }
}
