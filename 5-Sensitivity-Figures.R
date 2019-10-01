library(tidyverse)
library(feather)

mdat2km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_2k_cross_val_dat.feather')

mdat5km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_5k_cross_val_dat.feather')

mmdat <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_min_model_cross_val_dat.feather')


# ------------------------------------------------------------------------------------
# 5km Cross-validation results

# Figure ***
mdat2km$year_label <- paste0(mdat2km$year, " - F1: ", round(mdat2km$f1, 2), " AUC: ", round(mdat2km$auc, 2), " AP: ", round(mdat2km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mdat2km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision") +
  theme_tufte(12) +
  # scale_color_manual(values = viridis(6, option = "D")) +
  scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.225, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/2km_precision_recall.pdf", width = 6, height = 4)


# ------------------------------------------------------------------------------------
# 10km Cross-validation results

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
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.225, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/5km_precision_recall.pdf", width = 6, height = 4)





# ------------------------------------------------------------------------------------
# Minimal Model Cross-validation results

# Figure ***
mmdat$year_label <- paste0(mmdat$year, " - F1: ", round(mmdat$f1, 2), " AUC: ", round(mmdat$auc, 2), " AP: ", round(mmdat$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mmdat, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision") +
  theme_tufte(12) +
  # scale_color_manual(values = viridis(6, option = "D")) +
  scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.225, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/min_mod_precision_recall.pdf", width = 6, height = 4)