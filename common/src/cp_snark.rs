use crate::{model::Poly, Node, NodeCommitment};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

// In this trait, we closely follow the notation of the LegoSNARK paper. This
// trait should be used by the composition engine, but not directly implemented

// On PCS genericity. The fact that this trait is generic on the PCS is a
// consequence of our design needs: either the closely related trait
// NodeCPSNARK or the node structs themselves need to be generic on the PCS so
// that we can implement the latter trait for the nodes for any choice of a
// PCS. Genericity on the PCS and related params was deemed to be a good design
// choice, which only leaves us with the option of making NodeCPSNARK generic.
// In turn, this genericity forces CPSNARK itself to be generic on the PCS due
// to our blanket implementation of CPSNARK for all implementors of
// NodeCPSNARK.
//
// It should be stressed that the PCS this trait is generic on is *not* the
// commitment scheme from the definition of a CP-SNARK in the LegoSNARK paper.
// Indeed, commitment schemes and polynomial commitment schemes are different,
// incompatible cryptographic primitives.
//
// Note that the PCS does not appear in the associated types or methods of this
// trait. If this trait needs to be implemented for a struct which is genuinely
// unrelated to any PCS (which is never the case for our node-proving SNARKS),
// a dummy PCS should be used for the generic type.
trait CPSNARK<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    const ARITY: usize;

    type CommitmentKey; // ck used to commit and open commitments
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type Commitment; // C
    type Opening; // O (a.k.a. opening hint)

    type Instance; // D_x
    type Value; // D containing D_i for 1 <= i <= arity
    type Witness; // D_omega

    type Proof; // pi

    // A description of a relation on Instance x (Value x Witness)
    // which is used to generate the evaluation and verification keys
    type Relation;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        values: [&Self::Value; Self::ARITY],
        openings: [&Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof;

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool;
}

// Taylored version of CPSNARK adapted to node proofs
// This is the trait that should be implemented by each node struct
pub trait NodeCPSNARK<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type CommitmentKey; // ck used to commit and open commitments to parameters, inputs and outputs
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used to verify proofs

    type ParamCommitment; // C_1
    type ParamHint; // O_1 (a.k.a. opening hint)
    type ParamValue; // D_1

    type Instance; // D_x

    type Proof; // pi

    fn key_gen(
        ck: &Self::CommitmentKey,
        node: &Self,
    ) -> (Self::EvaluationKey, Self::VerificationKey);

    fn prove(
        ek: &Self::EvaluationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &PCS::Commitment,
        output_commitment: &PCS::Commitment,
        param_value: &Self::ParamValue,
        input_value: &Vec<F>,
        output_value: &Vec<F>,
        param_hint: &Self::ParamHint,
        input_hint: &PCS::CommitmentState,
        output_hint: &PCS::CommitmentState,
        // no non-committed witness omega
    ) -> Self::Proof;

    fn verify_proof(
        vk: &Self::VerificationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &PCS::Commitment,
        output_commitment: &PCS::Commitment,
        pi: &Self::Proof,
    ) -> bool;
}

enum ParamOrIO<ParamType, IOType> {
    Param(ParamType),
    IO(IOType),
}

fn get_param_input_output<ParamType, IOType>(
    array: [&ParamOrIO<ParamType, IOType>; 3],
) -> (&ParamType, &IOType, &IOType) {
    match array {
        [ParamOrIO::Param(p), ParamOrIO::IO(i), ParamOrIO::IO(o)] => (p, i, o),
        _ => panic!("Invalid structure"),
    }
}

impl<F, S, PCS, NCPS> CPSNARK<F, S, PCS> for NCPS
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    NCPS: NodeCPSNARK<F, S, PCS>,
{
    const ARITY: usize = 3;

    type CommitmentKey = NCPS::CommitmentKey;
    type EvaluationKey = NCPS::EvaluationKey;
    type VerificationKey = NCPS::VerificationKey;

    type Commitment = ParamOrIO<NCPS::ParamCommitment, PCS::Commitment>;
    type Opening = ParamOrIO<NCPS::ParamHint, PCS::CommitmentState>;
    type Value = ParamOrIO<NCPS::ParamValue, Vec<F>>;

    type Instance = NCPS::Instance;

    type Witness = ();

    type Proof = NCPS::Proof;

    type Relation = Self;

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: &Self::Relation,
    ) -> (Self::EvaluationKey, Self::VerificationKey) {
        NCPS::key_gen(ck, r)
    }

    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        values: [&Self::Value; Self::ARITY],
        openings: [&Self::Opening; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof {
        let (param_commitment, input_commitment, output_commitment) =
            get_param_input_output(commitments);
        let (param_value, input_value, output_value) = get_param_input_output(values);
        let (param_hint, input_hint, output_hint) = get_param_input_output(openings);

        NCPS::prove(
            ek,
            x,
            param_commitment,
            input_commitment,
            output_commitment,
            param_value,
            input_value,
            output_value,
            param_hint,
            input_hint,
            output_hint,
        )
    }

    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: [&Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool {
        let (param_commitment, input_commitment, output_commitment) =
            get_param_input_output(commitments);

        NCPS::verify_proof(
            vk,
            x,
            param_commitment,
            input_commitment,
            output_commitment,
            pi,
        )
    }
}
