suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(argparse)
  library(taproom)
})
options(warn=-1)

parser <- ArgumentParser(description='Input .rds file to be processed')
parser$add_argument('rds_in', type='character', help='input rds file')
parser$add_argument('csv_out', type='character', help='output csv file')
parser$add_argument('--assay', type='character', help='assay')
parser$add_argument('--design_col', type='character', help='design column')
parser$add_argument('--design_csv', type='character', help='output design csv file')
parser$add_argument('--winsorize', type='character', help="percentage remaining after winsorizing")
args <- parser$parse_args()

sce <- readRDS(args$rds_in)

win = as.numeric(args$winsorize)
for (channel in rownames(sce)) {
  sce[channel, ] = winsorize(sce[channel, ], exprs_values=args$assay, w_limits=c((1-win)/2, (1+win)/2))
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
