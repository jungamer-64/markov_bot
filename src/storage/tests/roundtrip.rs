use rand::{SeedableRng, rngs::StdRng};

use crate::{
    config::DynError, markov::GenerationOptions, storage::load_chain, test_support::ensure_eq,
};

use super::helpers::{run_async_test, sample_chain_with_order, write_sample_file};

#[test]
fn round_trips_multiple_ngram_orders() -> Result<(), DynError> {
    run_async_test(async {
        for order in [1_usize, 3, 6, 7, 16] {
            let chain = sample_chain_with_order(order)?;
            let path = write_sample_file("roundtrip_orders", &chain).await?;
            let loaded = load_chain(&path, order).await?;

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
    })
}

#[test]
fn round_trip_preserves_generation_for_seeded_rng() -> Result<(), DynError> {
    run_async_test(async {
        for order in [3_usize, 7, 16] {
            let chain = sample_chain_with_order(order)?;
            let path = write_sample_file("roundtrip_generation", &chain).await?;
            let loaded = load_chain(&path, order).await?;

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
    })
}
