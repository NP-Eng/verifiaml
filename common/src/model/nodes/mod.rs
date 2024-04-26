pub(crate) mod bmm;
pub(crate) mod relu;
pub(crate) mod requantize_bmm_float;
pub(crate) mod requantize_bmm_ref;
pub(crate) mod requantize_bmm_single;
pub(crate) mod reshape;

use std::any::Any;

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::{CryptographicSponge, Poly};

use self::{
    bmm::{BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof},
    requantize_bmm_float::{
        RequantizeBMMNodeCommitment, RequantizeBMMNodeCommitmentState, RequantizeBMMNodeProof,
    },
    requantize_bmm_ref::{
        RequantizeBMMRefNodeCommitment, RequantizeBMMRefNodeCommitmentState,
        RequantizeBMMRefNodeProof,
    },
    requantize_bmm_single::{
        RequantizeBMMSingleNodeCommitment, RequantizeBMMSingleNodeCommitmentState,
        RequantizeBMMSingleNodeProof,
    },
};

use super::tensor::{NIOTensor, SmallNIO};

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub trait NodeOpsNative<ST>: AsAny<ST>
where
    ST: SmallNIO,
{
    fn assert_valid_input(&self, input_shape: &Vec<usize>) {
        assert_eq!(
            input_shape.len(),
            self.shape().0.len(),
            "Incorrect shape: {} expects a {}-dimensional input array",
            self.shape().0.len(),
            self.type_name()
        );

        assert_eq!(
            self.shape().0[0],
            input_shape[0],
            "Length mismatch: {} expects input with {} elements, got {} elements instead",
            self.type_name(),
            self.shape().0[0],
            input_shape[0]
        );
    }

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().1.iter().product()
    }

    /// Returns the shapes of the node's input and output tensors
    fn shape(&self) -> (Vec<usize>, Vec<usize>);

    /// Evaluate the node natively (without padding)
    /// TODO decide whether this method should stay on `NodeOps`, or maybe go to `NodeOpsSNARKVerify`
    fn evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST>;

    /// Returns the maximum number of variables of the MLEs committed to as part of
    /// this nodes's commitment.
    fn com_num_vars(&self) -> usize {
        0
    }

    fn type_name(&self) -> &'static str;
}

pub trait AsAny<ST> {
    fn as_any(&self) -> &dyn Any;
}

impl<ST: SmallNIO, T> AsAny<ST> for T
where
    T: NodeOpsNative<ST> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait NodeOpsPadded<ST>: NodeOpsNative<ST>
where
    ST: SmallNIO,
{
    fn assert_valid_padded_input(&self, input_shape: &Vec<usize>) {
        assert_eq!(
            input_shape.len(),
            self.padded_shape().0.len(),
            "Incorrect shape: {} expects a {}-dimensional input array",
            self.padded_shape().0.len(),
            self.type_name()
        );

        assert_eq!(
            self.padded_shape().0[0],
            input_shape[0],
            "Length mismatch: {} expects input with {} elements, got {} elements instead",
            self.type_name(),
            self.padded_shape().0[0],
            input_shape[0]
        );
    }

    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    // TODO we could apply next_power_of_two to self.shape() elementwise, but
    // I expect this to be less efficient since each implementor will likely
    // internally store padded_shape_log
    fn padded_shape_log(&self) -> Vec<usize>;

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> (Vec<usize>, Vec<usize>);

    /// The log of the number of output units of the padded node
    fn padded_num_units_log(&self) -> usize {
        self.padded_shape_log().iter().sum()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().0.iter().product()
    }

    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST>;
}

pub enum NodeProof<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    BMM(BMMNodeProof<F, S, PCS>),
    RequantizeBMM(RequantizeBMMNodeProof),
    RequantizeBMRef(RequantizeBMMRefNodeProof),
    RequantizeBMMSingle(RequantizeBMMSingleNodeProof),
    ReLU(()),
    Reshape(()),
}

pub enum NodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    BMM(BMMNodeCommitment<F, S, PCS>),
    RequantizeBMM(RequantizeBMMNodeCommitment),
    RequantizeBMMRef(RequantizeBMMRefNodeCommitment),
    RequantizeBMMSingle(RequantizeBMMSingleNodeCommitment),
    ReLU(()),
    Reshape(()),
}

pub enum NodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    BMM(BMMNodeCommitmentState<F, S, PCS>),
    RequantizeBMM(RequantizeBMMNodeCommitmentState),
    RequantizeBMMRef(RequantizeBMMRefNodeCommitmentState),
    RequantizeBMMSingle(RequantizeBMMSingleNodeCommitmentState),
    ReLU(()),
    Reshape(()),
}
