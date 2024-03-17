use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::rand::RngCore;

use hcs_common::{
    InnerType, LabeledPoly, Node, NodeCommitment, NodeCommitmentState, NodeProof, Poly, QArray,
    QTypeArray,
};

mod model;
mod nodes;

pub use model::ProveModel;

pub trait NodeOpsProve<F, S, PCS, I, O>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Produce a node output proof
    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS>;

    /// Commit to the node parameters
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>);
}

/// Padded evaluation which each of the node types must implement.
pub trait NodeOpsPaddedEvaluate<I, O> {
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QArray<I>) -> QArray<O>;
}

/// We don't want:
/// `fn padded_evaluate(&self, input: &QArray<I>) -> QArray<O>;`
/// but instead:
/// `fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O>;`
/// so that we can have polymorphism and iterate over the different nodes, such that
/// the output type of the `padded_evaluate`` method is the same as the input type
/// of the next node in the model.
///
/// We cannot directly implement a new method `padded_evaluate` on a foreign enum `Node`.
/// Instead, we create a private wrapper trait to implement the desired method.
trait NodeOpsPaddedEvaluateWrapper<I, O>
where
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
    fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O>;
}

macro_rules! node_operation {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            Node::BMM(node) => node.$method($($arg),*),
            Node::RequantiseBMM(node) => node.$method($($arg),*),
            Node::ReLU(node) => node.$method($($arg),*),
            Node::Reshape(node) => NodeOpsProve::<_, _, _, I, _>::$method(node, $($arg),*),
        }
    };
}

impl<I, O> NodeOpsPaddedEvaluateWrapper<I, O> for Node<I, O>
where
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
    /// Here we perform matching without sanity checks. By design, the input type of the
    /// next node in the model is the same as the output type of the current node,
    /// so hiccups should never occur.
    fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O> {
        match (self, input) {
            (Node::BMM(fc), QTypeArray::S(input)) => QTypeArray::L(fc.padded_evaluate(input)),
            (Node::RequantiseBMM(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.padded_evaluate(input))
            }
            (Node::ReLU(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            (Node::Reshape(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            _ => panic!("Invalid input type for node"),
        }
    }
}

impl<F, S, PCS, I, O> NodeOpsProve<F, S, PCS, I, O> for Node<I, O>
where
    F: PrimeField + Absorb + From<I> + From<O> + From<O>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        node_operation!(
            self,
            prove,
            ck,
            s,
            node_com,
            node_com_state,
            input,
            input_com,
            input_com_state,
            output,
            output_com,
            output_com_state
        )
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        node_operation!(self, commit, ck, rng)
    }
}
