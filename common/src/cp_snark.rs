// In this trait, we closely follow the notation of the LegoSNARK paper
trait CPSNARK {
    const ARITY: usize;

    type CommitmentKey; // ck used to commit and open commitments
    type EvaluationKey; // ek used to produce proofs
    type VerificationKey; // vk used rto verify proofs

    type Commitment; // C
    type OpeningType; // O (a.k.a. opening hint)

    type Instance; // D_x
    type Value; // D containing D_i for 1 <= i <= arity
    type Witness; // D_omega

    type Proof; // pi

    fn key_gen(
        ck: &Self::CommitmentKey,
        r: dyn Fn(&Self::Instance, &[Self::Value; Self::ARITY], &Self::Witness) -> bool,
    ) -> (Self::EvaluationKey, Self::VerificationKey);
    fn prove(
        ek: &Self::EvaluationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        values: &[Self::Value; Self::ARITY],
        omega: &Self::Witness,
    ) -> Self::Proof;
    fn verify_proof(
        vk: &Self::VerificationKey,
        x: &Self::Instance,
        commitments: &[Self::Commitment; Self::ARITY],
        pi: &Self::Proof,
    ) -> bool;
}
