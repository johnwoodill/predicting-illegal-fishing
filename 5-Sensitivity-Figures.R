library(tidyverse)
library(feather)

mdat2km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_2k_cross_val_dat.feather')

mdat5km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_5k_cross_val_dat.feather')

mdat10km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_10k_cross_val_dat.feather')

mmdat <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_min_model_cross_val_dat.feather')


# ------------------------------------------------------------------------------------
# 2km Cross-validation results

# Figure ***
mdat2km$year_label <- paste0(mdat2km$year, " - F1: ", round(mdat2km$f1, 2), " AUC: ", round(mdat2km$auc, 2), " AP: ", round(mdat2km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p1 <- ggplot(mdat2km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="2 km EEZ Buffer") +
  theme_tufte(12) +
  scale_color_manual(values = viridis(6, option = "D")) +
  # scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.310, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.title.align = 0.5,
        # legend.text = element_text(size=8),
        # legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p1

# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1a.pdf", width = 6, height = 4)
# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1a.png", width = 6, height = 4)


# ------------------------------------------------------------------------------------
# 5km Cross-validation results

# Figure ***
mdat5km$year_label <- paste0(mdat5km$year, " - F1: ", round(mdat5km$f1, 2), " AUC: ", round(mdat5km$auc, 2), " AP: ", round(mdat5km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p2 <- ggplot(mdat5km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="5 km EEZ Buffer") +
  theme_tufte(12) +
  scale_color_manual(values = viridis(6, option = "D")) +
  # scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.310, .25),
        legend.direction = 'vertical',
        legend.title.align = 0.5,
        legend.justification = 'center',
        # legend.text = element_text(size=8),
        # legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p2

# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1b.pdf", width = 6, height = 4)
# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1b.png", width = 6, height = 4)



# ------------------------------------------------------------------------------------
# 10km Cross-validation results

# Figure ***
mdat10km$year_label <- paste0(mdat10km$year, " - F1: ", round(mdat10km$f1, 2), " AUC: ", round(mdat10km$auc, 2), " AP: ", round(mdat10km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p3 <- ggplot(mdat10km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="10 km EEZ Buffer") +
  theme_tufte(12) +
  scale_color_manual(values = viridis(6, option = "D")) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.310, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.title.align = 0.5,
        # legend.text = element_text(size=8),
        # legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p3

# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1c.pdf", width = 6, height = 4)
# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1c.png", width = 6, height = 4)



# ------------------------------------------------------------------------------------
# Minimal Model Cross-validation results

# Figure ***
mmdat$year_label <- paste0(mmdat$year, " - F1: ", round(mmdat$f1, 2), " AUC: ", round(mmdat$auc, 2), " AP: ", round(mmdat$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p4 <- ggplot(mmdat, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="Minimum RF Model") +
  theme_tufte(12) +
  scale_color_manual(values = viridis(6, option = "D")) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.310, .25),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.title.align = 0.5,
        #legend.text = element_text(size=8),
        #legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p4

# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1d.pdf", width = 6, height = 4)
# ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1d.png", width = 6, height = 4)

plot_grid(p4, p1, p2, p3, ncol=2, labels = c("(A)", "(B)", "(C)", "(D)"), label_x = .89, label_y = .98)

ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1.pdf", width = 10, height = 8)
ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1.png", width = 10, height = 8)
