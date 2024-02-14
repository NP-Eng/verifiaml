use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;

use crate::{
    model::{
        nodes::{fc::FCNode, relu::ReLUNode},
        CryptographicSponge, Poly,
    },
    quantization::QSmallType,
};

use self::{
    fc::{FCNodeCommitment, FCNodeCommitmentState, FCNodeProof},
    loose_fc::{LooseFCNode, LooseFCNodeCommitment, LooseFCNodeCommitmentState, LooseFCNodeProof},
    reshape::ReshapeNode,
};

use super::qarray::QArray;

pub(crate) mod fc;
pub(crate) mod loose_fc;
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
pub(crate) trait NodeOps {
    /// Returns the shape of the node's output tensor
    fn shape(&self) -> Vec<usize>;

    /// The number of output units of the node
    fn num_units(&self) -> usize {
        self.shape().iter().product()
    }

    /// Evaluate the node natively (without padding)
    fn evaluate(&self, input: &QArray<QSmallType>) -> QArray<QSmallType>;
}

pub(crate) trait NodeOpsSNARK<F, S, PCS>
where
    F: PrimeField,
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

    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QArray<QSmallType>) -> QArray<QSmallType>;

    /// Commit to the node parameters
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>);

    /// Produce a node output proof
    fn prove(
        &self,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input: QArray<QSmallType>,
        input_com: &PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: &PCS::Commitment,
    ) -> NodeProof;
}

pub(crate) enum Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNode<F, S, PCS>),
    LooseFC(LooseFCNode<F, S, PCS>),
    ReLU(ReLUNode<F, S, PCS>),
    Reshape(ReshapeNode<F, S, PCS>),
}

pub(crate) enum NodeProof {
    FC(FCNodeProof),
    LooseFC(LooseFCNodeProof),
    ReLU(()),
    Reshape(()),
}

pub(crate) enum NodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNodeCommitment<F, S, PCS>),
    LooseFC(LooseFCNodeCommitment<F, S, PCS>),
    ReLU(()),
    Reshape(()),
}

pub(crate) enum NodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    FC(FCNodeCommitmentState<F, S, PCS>),
    LooseFC(LooseFCNodeCommitmentState<F, S, PCS>),
    ReLU(()),
    Reshape(()),
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> Node<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn as_node_ops(&self) -> &dyn NodeOps {
        match self {
            Node::FC(fc) => fc,
            Node::LooseFC(fc) => fc,
            Node::ReLU(r) => r,
            Node::Reshape(r) => r,
        }
    }

    fn as_node_ops_snark(&self) -> &dyn NodeOpsSNARK<F, S, PCS> {
        match self {
            Node::FC(fc) => fc,
            Node::LooseFC(fc) => fc,
            Node::ReLU(r) => r,
            Node::Reshape(r) => r,
        }
    }

    // Print the type of the node. This cannot be cleantly achieved by deriving
    // Debug
    fn type_name(&self) -> &'static str {
        match self {
            Node::FC(_) => "FC",
            Node::LooseFC(_) => "LooseFC",
            Node::ReLU(_) => "ReLU",
            Node::Reshape(_) => "Reshape",
        }
    }
}
// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> NodeOps for Node<F, S, PCS>
where
    F: PrimeField,
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
    fn evaluate(&self, input: &QArray<QSmallType>) -> QArray<QSmallType> {
        self.as_node_ops().evaluate(input)
    }
}

impl<F, S, PCS> NodeOpsSNARK<F, S, PCS> for Node<F, S, PCS>
where
    F: PrimeField,
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

    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QArray<QSmallType>) -> QArray<QSmallType> {
        self.as_node_ops_snark().padded_evaluate(input)
    }

    /// Commit to the node parameters
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        self.as_node_ops_snark().commit(ck, rng)
    }

    fn prove(
        &self,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input: QArray<QSmallType>,
        input_com: &PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: &PCS::Commitment,
    ) -> NodeProof {
        self.as_node_ops_snark()
            .prove(s, node_com, input, input_com, output, output_com)
    }
}
