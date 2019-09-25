library(tidyverse)
library(feather)

mdat5km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_5k_cross_val_dat.feather')

mdat10km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_10k_cross_val_dat.feather')




# ------------------------------------------------------------------------------------
# 5km Cross-validation results

# Figure ***
mdat5km$year_label <- paste0(mdat5km$year, " - F1: ", round(mdat5km$f1, 2), " AUC: ", round(mdat5km$auc, 2), " AP: ", round(mdat5km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mdat5km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision") +
  theme_tufte(12) +
  # scale_color_manual(values = viridis(6, option = "D")) +
  scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.position = c(.20, .20),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/5km_precision_recall.pdf", width = 6, height = 4)


# ------------------------------------------------------------------------------------
# 10km Cross-validation results

# Figure ***
mdat10km$year_label <- paste0(mdat10km$year, " - F1: ", round(mdat10km$f1, 2), " AUC: ", round(mdat10km$auc, 2), " AP: ", round(mdat10km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mdat10km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision") +
  theme_tufte(12) +
  # scale_color_manual(values = viridis(6, option = "D")) +
  scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.position = c(.20, .20),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/5km_precision_recall.pdf", width = 6, height = 4)
