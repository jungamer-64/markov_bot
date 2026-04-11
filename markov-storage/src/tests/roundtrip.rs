use rand::{SeedableRng, rngs::StdRng};

use markov_core::GenerationOptions;

use super::helpers::{ensure_eq, load_sample_file, sample_chain_with_order, write_sample_file};

#[test]
fn round_trips_multiple_ngram_orders() -> Result<(), crate::StorageError> {
    for order in [1_usize, 3, 6, 7, 16] {
        let chain = sample_chain_with_order(order)?;
        let path = write_sample_file("roundtrip_orders", &chain)?;
        let loaded = load_sample_file(&path, order)?;

        ensure_eq(&loaded.ngram_order, &order, "ngram order should round-trip")?;
        ensure_eq(
            &loaded.models.len(),
            &order,
            "model section count should match ngram order",
        )?;
        ensure_eq(
            &loaded.id_to_token,
            &chain.id_to_token,
            "vocabulary should round-trip",
        )?;
        ensure_eq(
            &loaded.starts,
            &chain.starts,
            "start prefixes should round-trip",
        )?;
        ensure_eq(&loaded.models, &chain.models, "models should round-trip")?;
    }

    Ok(())
}

#[test]
fn round_trip_preserves_generation_for_seeded_rng() -> Result<(), crate::StorageError> {
    for order in [3_usize, 7, 16] {
        let chain = sample_chain_with_order(order)?;
        let path = write_sample_file("roundtrip_generation", &chain)?;
        let loaded = load_sample_file(&path, order)?;

        let mut original_rng = StdRng::seed_from_u64(1234);
        let mut loaded_rng = StdRng::seed_from_u64(1234);
        let options = GenerationOptions::new(12, 1.0, 0);

        let original = chain.generate_sentence_with_options(&mut original_rng, options);
        let rebuilt = loaded.generate_sentence_with_options(&mut loaded_rng, options);

        ensure_eq(
            &rebuilt,
            &original,
            "loaded chain should generate the same seeded sentence",
        )?;
    }

    Ok(())
}
