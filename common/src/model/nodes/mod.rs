use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::{
    nodes::{bmm::BMMNode, relu::ReLUNode},
    CryptographicSponge, Poly,
};

use self::{
    bmm::{BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof},
    requantise_bmm::{
        RequantiseBMMNode, RequantiseBMMNodeCommitment, RequantiseBMMNodeCommitmentState,
        RequantiseBMMNodeProof,
    },
    reshape::ReshapeNode,
};

use super::qarray::QTypeArray;

pub(crate) mod bmm;
pub(crate) mod relu;
pub(crate) mod requantise_bmm;
pub(crate) mod reshape;

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub(crate) trait NodeOpsNative {
    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize>;

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// Evaluate the node natively (without padding)
    /// TODO decide whether this method should stay on `NodeOps`, or maybe go to `NodeOpsSNARKVerify`
    fn evaluate(&self, input: &QTypeArray) -> QTypeArray;
}

pub trait NodeOpsCommon<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
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
}

pub enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    BMM(BMMNode<F, S, PCS>),
    RequantiseBMM(RequantiseBMMNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
    Reshape(ReshapeNode<F, S, PCS>),
}

pub enum NodeProof<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    BMM(BMMNodeProof<F, S, PCS>),
    RequantiseBMM(RequantiseBMMNodeProof),
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
    RequantiseBMM(RequantiseBMMNodeCommitment),
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
    RequantiseBMM(RequantiseBMMNodeCommitmentState),
    ReLU(()),
    Reshape(()),
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn as_node_ops(&self) -> &dyn NodeOpsNative {
        match self {
            Node::BMM(fc) => fc,
            Node::RequantiseBMM(r) => r,
            Node::ReLU(r) => r,
            Node::Reshape(r) => r,
        }
    }

    pub fn as_node_ops_snark(&self) -> &dyn NodeOpsCommon<F, S, PCS> {
        match self {
            Node::BMM(fc) => fc,
            Node::RequantiseBMM(r) => r,
            Node::ReLU(r) => r,
            Node::Reshape(r) => r,
        }
    }

    // Print the type of the node. This cannot be cleantly achieved by deriving
    // Debug
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Node::BMM(_) => "BMM",
            Node::RequantiseBMM(_r) => "RequantiseBMM",
            Node::ReLU(_) => "ReLU",
            Node::Reshape(_) => "Reshape",
        }
    }
}
// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> NodeOpsNative for Node<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize> {
        self.as_node_ops().shape()
    }

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.as_node_ops().num_units()
    }

    /// Evaluate the node natively (without padding)
    fn evaluate(&self, input: &QTypeArray) -> QTypeArray {
        self.as_node_ops().evaluate(input)
    }
}

impl<F, S, PCS> NodeOpsCommon<F, S, PCS> for Node<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    fn padded_shape_log(&self) -> Vec<usize> {
        self.as_node_ops_snark().padded_shape_log()
    }

    fn com_num_vars(&self) -> usize {
        self.as_node_ops_snark().com_num_vars()
    }
}
