# Regenerate R-equivalence fixtures for longcombat-py.
#
# Usage (from the repository root):
#
#     Rscript tests/fixtures/regenerate.R
#
# This script:
#   1. Builds the same synthetic dataset used by tests/conftest.py
#      ``synthetic_data`` fixture (seed 1, 30 subjects × 4 visits,
#      3 features, batches A/B/C with known additive and multiplicative
#      effects).
#   2. Runs jcbeer/longCombat on it.
#   3. Writes four CSV fixtures into tests/fixtures/:
#        r_input.csv, r_data_combat.csv, r_gammahat.csv, r_delta2hat.csv,
#        r_gammastarhat.csv, r_delta2starhat.csv
#
# Requires R packages: lme4, longCombat (install from GitHub with devtools).

suppressPackageStartupMessages({
  library(lme4)
  library(longCombat)
})

set.seed(1)

# ------ generate the dataset (must match tests/conftest.py) ---------------
make_synthetic <- function() {
  n_subj <- 30L
  n_visit <- 4L
  n_feat <- 3L
  L <- n_subj * n_visit
  subid <- rep(seq_len(n_subj) - 1L, each = n_visit)
  time <- rep(seq_len(n_visit) - 1L, times = n_subj)
  age <- rep(sample(20:59, n_subj, replace = TRUE), each = n_visit)
  dx <- rep(sample(0:1, n_subj, replace = TRUE), each = n_visit)
  batch <- sample(c("A", "B", "C"), L, replace = TRUE)
  subj_re <- rnorm(n_subj, 0, 1)
  batch_add <- c(A = 0.0, B = 2.0, C = -1.5)
  batch_mult <- c(A = 1.0, B = 0.5, C = 1.5)

  df <- data.frame(
    subid = subid, time = time, age = age, diagnosis = dx, batch = batch
  )
  for (v in seq_len(n_feat)) {
    noise <- rnorm(L, 0, 1)
    signal <- 0.01 * age - 0.1 * time + 0.5 * dx - 0.2 * dx * time +
      subj_re[subid + 1L]
    bmul <- batch_mult[batch]; badd <- batch_add[batch]
    df[[paste0("feat", v)]] <- signal + bmul * noise + badd
  }
  df
}

# NOTE: The RNG stream in Python's numpy.random.default_rng(1) and R's
# set.seed(1) are NOT identical. This script therefore regenerates the
# dataset deterministically from R's RNG and saves the *actual inputs* it
# used alongside the outputs — the Python test reads r_input.csv rather
# than rebuilding from seed. That way we compare Python's long_combat on
# the R-generated data to R's longCombat on the same data.

df <- make_synthetic()

# Resolve the output directory: try (in order) the parent of this script
# (works when invoked as `Rscript tests/fixtures/regenerate.R`), then the
# working directory, then `./tests/fixtures`.
resolve_out_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
    if (dir.exists(script_dir)) return(script_dir)
  }
  candidate <- file.path(getwd(), "tests", "fixtures")
  if (dir.exists(candidate)) return(candidate)
  getwd()
}
out_dir <- resolve_out_dir()
message("Writing fixtures to: ", normalizePath(out_dir))
write.csv(df, file.path(out_dir, "r_input.csv"), row.names = FALSE)

# ------ run longCombat ----------------------------------------------------
res <- longCombat(
  idvar = "subid",
  timevar = "time",
  batchvar = "batch",
  features = paste0("feat", 1:3),
  formula = "age + diagnosis*time",
  ranef = "(1|subid)",
  data = df,
  niter = 30,
  method = "REML",
  verbose = FALSE
)

write.csv(res$data_combat, file.path(out_dir, "r_data_combat.csv"),
          row.names = FALSE)
write.csv(as.data.frame(res$gammahat), file.path(out_dir, "r_gammahat.csv"),
          row.names = TRUE)
write.csv(as.data.frame(res$delta2hat), file.path(out_dir, "r_delta2hat.csv"),
          row.names = TRUE)
write.csv(as.data.frame(res$gammastarhat),
          file.path(out_dir, "r_gammastarhat.csv"), row.names = TRUE)
write.csv(as.data.frame(res$delta2starhat),
          file.path(out_dir, "r_delta2starhat.csv"), row.names = TRUE)

message("Done.")
