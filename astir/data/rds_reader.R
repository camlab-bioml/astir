suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(argparse)
})

parser <- ArgumentParser(description='Input .rds file to be processed')
parser$add_argument('rds_in', type='character', help='input rds file')
parser$add_argument('csv_out', type='character', help='output csv file')
parser$add_argument('--assay', type='character', help='assay')
parser$add_argument('--design_col', type='character', help='design column')
parser$add_argument('--design_csv', type='character', help='output design csv file')
args <- parser$parse_args()

sce <- readRDS(args$rds_in)
# print(sce)
sce_df <- t(data.frame(assay(sce, args$assay)))
write.csv(sce_df, args$csv_out, row.names=TRUE)

design = args$design_col
if (!is.null(design)) {
  mm <- as.data.frame(colData(sce))
  ind = which(colnames(mm) == design)
  colnames(mm)[ind] <- 'target_col'
  # print(head(mm))
  design_mm <- model.matrix(~target_col, mm)
  write.csv(design_mm, args$design_csv, row.names=TRUE)
}
