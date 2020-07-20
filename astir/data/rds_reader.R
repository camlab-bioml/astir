suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(argparse)
})
options(warn=-1)

parser <- ArgumentParser(description='Input .rds file to be processed')
parser$add_argument('rds_in', type='character', help='input rds file')
parser$add_argument('csv_out', type='character', help='output csv file')
parser$add_argument('--assay', type='character', help='assay')
parser$add_argument('--design_col', type='character', help='design column')
parser$add_argument('--design_csv', type='character', help='output design csv file')
parser$add_argument('--winsorize', type='character', help='the winsorize limit will be c(<win>, 1-<win>)')
args <- parser$parse_args()

sce <- readRDS(args$rds_in)

winsorize <- function(sce,
                      exprs_values = "logcounts",
                      w_limits = c(0.05, 0.95)) {
  ## Save unwinsorized expression values
  assay(sce, paste0(exprs_values, "_unwinsorized")) <- assay(sce, exprs_values)
  
  assay(sce, exprs_values) <- t(apply(assay(sce, exprs_values),
                                      1,
                                      winsorize_one,
                                      w_limits))
  sce
}

win = as.numeric(args$winsorize)
for (channel in rownames(sce)) {
  sce[channel, ] = winsorize(sce[channel, ], exprs_values=args$assay, w_limits=c(win, 1-win))
}

# print(sce)
sce_df <- t(data.frame(assay(sce, args$assay)))
write.csv(sce_df, args$csv_out, row.names=TRUE)

design = args$design_col
if (design != "") {
  mm <- as.data.frame(colData(sce))
  ind = which(colnames(mm) == design)
  colnames(mm)[ind] <- 'target_col'
  # print(head(mm))
  design_mm <- model.matrix(~target_col, mm)
  if (args$design_csv != "") {
    write.csv(design_mm, args$design_csv, row.names=TRUE)
  }
}
