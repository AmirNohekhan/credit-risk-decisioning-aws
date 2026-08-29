# Default definition and temporal design

The primary target is **12-month PD** for an unsecured personal loan. The observation date is the application/origination date; every model feature must be known then. Default is the first occurrence within the next 12 months of 90+ days past due or charge-off. Bankruptcy would be included in production when reliably sourced, but is not simulated separately.

A loan with a default before month 12 is immediately classifiable. Otherwise it becomes a non-default only when the complete 12-month performance window has elapsed. Recent non-defaulting loans are censored and excluded, never relabeled as good. Prepayment without prior default is non-default, provided its observed contractual outcome is complete. Model development uses chronological origination splits; outcome records and recoveries are excluded from PD features.

Monthly scheduled and actual payment, balance, DPD, charge-off, prepayment, and recovery records derive the target. The simulator directly retains a latent counterfactual outcome solely to demonstrate acceptance bias; that field is not a model feature and would not exist in real production data.

