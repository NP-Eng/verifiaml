pub(crate) mod bmm;
pub(crate) mod relu;
pub(crate) mod requantize_bmm_float;
pub(crate) mod requantize_bmm_ref;
pub(crate) mod requantize_bmm_single;
pub(crate) mod reshape;

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{
    model::{
        nodes::{bmm::BMMNode, relu::ReLUNode},
        CryptographicSponge, Poly,
    },
    Tensor,
};

use self::{
    bmm::{BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof},
    requantize_bmm_float::{
        RequantizeBMMFloatNode, RequantizeBMMNodeCommitment, RequantizeBMMNodeCommitmentState,
        RequantizeBMMNodeProof,
    },
    requantize_bmm_ref::{
        RequantizeBMMRefNode, RequantizeBMMRefNodeCommitment, RequantizeBMMRefNodeCommitmentState,
        RequantizeBMMRefNodeProof,
    },
    requantize_bmm_single::{
        RequantizeBMMSingleNode, RequantizeBMMSingleNodeCommitment,
        RequantizeBMMSingleNodeCommitmentState, RequantizeBMMSingleNodeProof,
    },
    reshape::ReshapeNode,
};

use super::tensor::{Integral, QTypeArray};

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub trait NodeOpsNative<I, O> {
    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize>;

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// Evaluate the node natively (without padding)
    /// TODO decide whether this method should stay on `NodeOps`, or maybe go to `NodeOpsSNARKVerify`
    fn evaluate(&self, input: &Tensor<I>) -> Tensor<O>;
}

pub trait NodeOpsPadded<I, O>: NodeOpsNative<I, O> {
    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    // TODO we could apply next_power_of_two to self.shape() elementwise, but
    // I expect this to be less efficient since each implementor will likely
    // internally store padded_shape_log
    fn padded_shape_log(&self) -> Vec<usize>;

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log()
            .into_iter()
            .map(|x| 1 << x)
            .collect()
    }

    /// The log of the number of output units of the padded node
    fn padded_num_units_log(&self) -> usize {
        self.padded_shape_log().iter().sum()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    /// Returns the maximum number of variables of the MLEs committed to as part of
    /// this nodes's commitment.
    fn com_num_vars(&self) -> usize;

    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &Tensor<I>) -> Tensor<O>;
}

pub enum Node<ST, LT> {
    BMM(BMMNode<ST, LT>),
    RequantizeBMMFloat(RequantizeBMMFloatNode<ST>),
    RequantizeBMMRef(RequantizeBMMRefNode<ST, LT>),
    RequantizeBMMSingle(RequantizeBMMSingleNode<ST, LT>),
    ReLU(ReLUNode<ST>),
    Reshape(ReshapeNode),
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

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<I, O> Node<I, O>
where
    I: Integral + TryFrom<O>,
    O: Integral + From<I>,
{
    // Print the type of the node. This cannot be cleantly achieved by deriving
    // Debug
    pub fn type_name(&self) -> &'static str {
        match self {
            Node::BMM(_) => "BMM",
            Node::RequantizeBMMFloat(_r) => "RequantizeBMM",
            Node::RequantizeBMMRef(_r) => "RequantizeBMMRef",
            Node::RequantizeBMMSingle(_r) => "RequantizeBMMSingle",
            Node::ReLU(_) => "ReLU",
            Node::Reshape(_) => "Reshape",
        }
    }

    /// Returns the shape of the node's output tensor
    pub fn shape(&self) -> Vec<usize> {
        node_op!(self, shape, NodeOpsNative)
    }

    /// Evaluate the node natively (without padding)
    pub fn evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O> {
        match (self, input) {
            (Node::BMM(fc), QTypeArray::S(input)) => QTypeArray::L(fc.evaluate(input)),
            (Node::RequantizeBMMFloat(r), QTypeArray::L(input)) => QTypeArray::S(r.evaluate(input)),
            (Node::RequantizeBMMRef(r), QTypeArray::L(input)) => QTypeArray::S(r.evaluate(input)),
            (Node::RequantizeBMMSingle(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.evaluate(input))
            }
            (Node::ReLU(r), QTypeArray::S(input)) => QTypeArray::S(r.evaluate(input)),
            (Node::Reshape(r), QTypeArray::S(input)) => QTypeArray::S(r.evaluate(input)),
            _ => panic!(
                "Type mismatch: node of type {} received input of type {}",
                self.type_name(),
                input.variant_name()
            ),
        }
    }

    pub fn com_num_vars(&self) -> usize {
        node_op!(self, com_num_vars, NodeOpsPadded)
    }

    /// Here we perform matching without sanity checks. By design, the input type of the
    /// next node in the model is the same as the output type of the current node,
    /// so hiccups should never occur.
    pub fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O> {
        match (self, input) {
            (Node::BMM(fc), QTypeArray::S(input)) => QTypeArray::L(fc.padded_evaluate(input)),
            (Node::RequantizeBMMFloat(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.padded_evaluate(input))
            }
            (Node::RequantizeBMMRef(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.padded_evaluate(input))
            }
            (Node::RequantizeBMMSingle(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.padded_evaluate(input))
            }
            (Node::ReLU(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            (Node::Reshape(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            _ => panic!("Invalid input type for node"),
        }
    }
}
