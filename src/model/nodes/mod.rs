use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{model::{
    nodes::{fc::FCNode, relu::ReLUNode}, qarray::InnerType, CryptographicSponge, Poly
}, quantization::QSmallType};

use self::reshape::ReshapeNode;

use super::qarray::QArray;

pub(crate) mod fc;
pub(crate) mod relu;
pub(crate) mod reshape;

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub(crate) trait NodeOps<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// A commitment associated to the node's transition function. For
    /// instance, it is empty for a ReLU transition; and a commitment to the
    /// matrix and bias for a MatMul transition.
    type NodeCommitment;

    /// A proof of execution of the node's transition function to a particular
    /// set of node values
    type Proof;

    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize>;

    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    // TODO we could apply next_power_of_two to self.shape() elementwise, but
    // I expect this to be less efficient since each implementor will likely
    // internally store padded_shape_log
    fn padded_shape_log(&self) -> Vec<usize>;

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log().into_iter().map(|x| 1 << x).collect()
    }

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    /// Evaluate the node natively (without padding)
    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType>;

    // TODO: is it okay to trim all keys from the same original PCS key?
    // (e.g. to trim the same key to for the matrix and for the bias in the
    // case of MatMul)
    // fn setup(&self, params: PCS::UniversalParams) -> (, Self::VerifierKey);

    /// Commit to the node parameters
    fn commit(&self) -> Self::NodeCommitment;

    /// Produce a node output proof
    fn prove(
        node_com: Self::NodeCommitment,
        input: QArray<QSmallType>,
        input_com: PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: PCS::Commitment,
    ) -> Self::Proof;

    /// Verify a node output proof
    fn verify(node_com: Self::NodeCommitment, proof: Self::Proof) -> bool;
}

pub(crate) enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
    Reshape(ReshapeNode<F, S, PCS>),
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{

    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize> {
        match self {
            Node::FC(n) => n.shape(),
            Node::ReLU(r) => r.shape(),
            Node::Reshape(r) => r.shape(),
        }
    }

    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    fn padded_shape_log(&self) -> Vec<usize> {
        match self {
            Node::FC(n) => n.padded_shape_log(),
            Node::ReLU(r) => r.padded_shape_log(),
            Node::Reshape(r) => r.padded_shape_log(),
        }
    }

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log().into_iter().map(|x| 1 << x).collect()
    }

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    /// Evaluate the node natively (withotu padding)
    pub(crate) fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        match self {
            Node::FC(n) => n.evaluate(input),
            Node::ReLU(r) => r.evaluate(input),
            Node::Reshape(r) => r.evaluate(input),
        }
    }

    /// Commit to the node parameters
    // pub(crate) fn commit(&self) -> PCS::Commitment {
    //     match self {
    //         Node::FC(n) => n.commit(),
    //         Node::ReLU(r) => r.commit(),
    //         Node::Reshape(r) => r.commit(),
    //     }
    // }

    /// Produce a node output proof
    pub(crate) fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    /// Verify a node output proof
    pub(crate) fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }
}
