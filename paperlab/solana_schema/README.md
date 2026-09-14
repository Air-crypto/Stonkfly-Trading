Event definitions extracted from the MIT-licensed official @pump-fun/pump-sdk 2.0.0 package. The npm SHA-512 integrity was verified before publication. See pump_events.json for the package URL, integrity and complete source IDL hash. The 1.36.0 layout in the reference engine lacks the current creator/holder reward fields. Only these event layouts are included; no transaction-building or wallet code is imported.

`pump_pool_account.json` pins the official pump-public-docs PumpSwap IDL's Pool
account discriminator and type, with source URL and source SHA-256. The all-observed
collector verifies account owner/discriminator and reads the static identity prefix
to resolve existing pools. It never backfills reserve observations from account data.
