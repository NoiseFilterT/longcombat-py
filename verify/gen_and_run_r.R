# Independent verification harness (R side).
# Generates fresh longitudinal datasets in R, runs the REAL longCombat /
# addTest / multTest on them, and writes inputs + outputs to verify/out/.
# The Python side (run_py_compare.py) reads the SAME input CSVs and compares.

suppressPackageStartupMessages({
  library(lme4)
  library(longCombat)
})

OUT <- "verify/out"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

gen_data <- function(seed, n_subj, n_visit, n_feat, batches, ranslope = FALSE,
                     add_sd = 1.5, mult_lo = 0.5, mult_hi = 1.8, noise_sd = 1.0) {
  set.seed(seed)
  L <- n_subj * n_visit
  subid <- rep(seq_len(n_subj), each = n_visit)
  time <- rep(seq_len(n_visit) - 1L, times = n_subj)
  age <- rep(sample(20:70, n_subj, replace = TRUE), each = n_visit)
  dx <- rep(sample(0:1, n_subj, replace = TRUE), each = n_visit)
  # ensure every batch present at least twice: keep sampling until satisfied
  repeat {
    batch <- sample(batches, L, replace = TRUE)
    if (all(table(factor(batch, levels = batches)) >= 2)) break
  }
  re_int <- rnorm(n_subj, 0, 1.0)[subid]
  re_slope <- if (ranslope) rnorm(n_subj, 0, 0.3)[subid] else rep(0, L)
  badd <- setNames(rnorm(length(batches), 0, add_sd), batches)
  bmul <- setNames(runif(length(batches), mult_lo, mult_hi), batches)
  df <- data.frame(subid = subid, time = time, age = age,
                   diagnosis = dx, batch = batch)
  for (v in seq_len(n_feat)) {
    noise <- rnorm(L, 0, noise_sd)
    signal <- 0.02 * age - 0.1 * time + 0.5 * dx - 0.2 * dx * time +
      re_int + re_slope * time
    df[[paste0("feat", v)]] <- signal + bmul[batch] * noise + badd[batch]
  }
  df
}

scenarios <- list(
  s1_baseline = list(seed = 101, n_subj = 40, n_visit = 4, n_feat = 5,
                     batches = c("A", "B", "C"), ranef = "(1|subid)",
                     formula = "age + diagnosis*time", method = "REML",
                     ranslope = FALSE),
  s2_ranslope = list(seed = 202, n_subj = 60, n_visit = 5, n_feat = 5,
                     batches = c("A", "B", "C"), ranef = "(1 + time|subid)",
                     formula = "age + diagnosis*time", method = "REML",
                     ranslope = TRUE),
  s3_msr = list(seed = 303, n_subj = 40, n_visit = 4, n_feat = 5,
                batches = c("A", "B", "C"), ranef = "(1|subid)",
                formula = "age + diagnosis*time", method = "MSR",
                ranslope = FALSE),
  s4_6batch = list(seed = 404, n_subj = 90, n_visit = 4, n_feat = 6,
                   batches = c("b1", "b2", "b3", "b4", "b5", "b6"),
                   ranef = "(1|subid)", formula = "age + diagnosis*time",
                   method = "REML", ranslope = FALSE),
  s5_simpleformula = list(seed = 505, n_subj = 30, n_visit = 3, n_feat = 4,
                          batches = c("A", "B"), ranef = "(1|subid)",
                          formula = "time", method = "REML", ranslope = FALSE),
  s6_large = list(seed = 606, n_subj = 150, n_visit = 6, n_feat = 8,
                  batches = c("A", "B", "C", "D"), ranef = "(1|subid)",
                  formula = "age + diagnosis*time", method = "REML",
                  ranslope = FALSE)
)

run_one <- function(name, sc) {
  cat("=== scenario", name, "===\n")
  df <- gen_data(sc$seed, sc$n_subj, sc$n_visit, sc$n_feat, sc$batches,
                 ranslope = sc$ranslope)
  feats <- paste0("feat", seq_len(sc$n_feat))
  write.csv(df, file.path(OUT, paste0(name, "_input.csv")), row.names = FALSE)

  res <- longCombat(idvar = "subid", timevar = "time", batchvar = "batch",
                    features = feats, formula = sc$formula, ranef = sc$ranef,
                    data = df, niter = 30, method = sc$method, verbose = FALSE)
  write.csv(res$data_combat, file.path(OUT, paste0(name, "_data_combat.csv")),
            row.names = FALSE)
  write.csv(as.data.frame(res$gammahat), file.path(OUT, paste0(name, "_gammahat.csv")),
            row.names = TRUE)
  write.csv(as.data.frame(res$delta2hat), file.path(OUT, paste0(name, "_delta2hat.csv")),
            row.names = TRUE)
  write.csv(as.data.frame(res$gammastarhat), file.path(OUT, paste0(name, "_gammastarhat.csv")),
            row.names = TRUE)
  write.csv(as.data.frame(res$delta2starhat), file.path(OUT, paste0(name, "_delta2starhat.csv")),
            row.names = TRUE)

  # record params for the python side
  params <- list(name = name, ranef = sc$ranef, formula = sc$formula,
                 method = sc$method, features = feats)
  writeLines(jsonlite_min(params), file.path(OUT, paste0(name, "_params.json")))
  cat("  wrote outputs for", name, "\n")
}

# tiny JSON writer to avoid a jsonlite dependency
jsonlite_min <- function(p) {
  q <- function(x) paste0('"', x, '"')
  arr <- paste0("[", paste(sapply(p$features, q), collapse = ","), "]")
  paste0("{",
         q("name"), ":", q(p$name), ",",
         q("ranef"), ":", q(p$ranef), ",",
         q("formula"), ":", q(p$formula), ",",
         q("method"), ":", q(p$method), ",",
         q("features"), ":", arr,
         "}")
}

for (nm in names(scenarios)) run_one(nm, scenarios[[nm]])

# ---- addTest / multTest on the baseline scenario (KR + Fligner-Killeen) ----
cat("=== addTest / multTest on s1_baseline ===\n")
df1 <- read.csv(file.path(OUT, "s1_baseline_input.csv"))
feats1 <- paste0("feat", 1:5)
add_r <- addTest(idvar = "subid", batchvar = "batch", features = feats1,
                 formula = "age + diagnosis*time", ranef = "(1|subid)",
                 data = df1, verbose = FALSE)
write.csv(add_r, file.path(OUT, "s1_addtest_R.csv"), row.names = FALSE)
mult_r <- multTest(idvar = "subid", batchvar = "batch", features = feats1,
                   formula = "age + diagnosis*time", ranef = "(1|subid)",
                   data = df1, verbose = FALSE)
write.csv(mult_r, file.path(OUT, "s1_multtest_R.csv"), row.names = FALSE)
cat("addTest (R, Kenward-Roger):\n"); print(add_r)
cat("multTest (R, Fligner-Killeen):\n"); print(mult_r)

cat("\nALL R OUTPUTS WRITTEN to", OUT, "\n")
