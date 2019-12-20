library(tidyverse)
library(feather)
library(ggthemes)
library(viridis)

mdattop5 <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_top5_cross_val_dat.feather')

mdatbio <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_bio_cross_val_dat.feather')

mdat2km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_2k_cross_val_dat.feather')

mdat5km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_5k_cross_val_dat.feather')

mdat10km <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_10k_cross_val_dat.feather')




# ------------------------------------------------------------------------------------
# 2km Cross-validation results

# Figure ***
mdat2km$year_label <- paste0(mdat2km$year, " - F1: ", round(mdat2km$f1, 2), " AUC: ", round(mdat2km$auc, 2), " AP: ", round(mdat2km$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p1 <- ggplot(mdat2km, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="2 km EEZ Buffer") +
  theme_tufte(11) +
  scale_color_manual(values = viridis(6, option = "D")) +
  # scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        # legend.position = c(.310, .25),
        legend.position = c(.28, .40),
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
  theme_tufte(11) +
  scale_color_manual(values = viridis(6, option = "D")) +
  # scale_color_manual(values = cbp1) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        # legend.position = c(.310, .25),
        legend.position = c(.28, .40),
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
  theme_tufte(11) +
  scale_color_manual(values = viridis(6, option = "D")) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        # legend.position = c(.310, .25),
        legend.position = c(.28, .40),
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
# Top five variables

# Figure ***
mdattop5$year_label <- paste0(mdattop5$year, " - F1: ", round(mdattop5$f1, 2), " AUC: ", round(mdattop5$auc, 2), " AP: ", round(mdattop5$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p4 <- ggplot(mdattop5, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="Top 5 Variables") +
  theme_tufte(11) +
  scale_color_manual(values = viridis(6, option = "D")) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        # legend.position = c(.310, .25),
        legend.position = c(.28, .40),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.title.align = 0.5,
        #legend.text = element_text(size=8),
        #legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p4



# ------------------------------------------------------------------------------------
# Bio model 

# Figure ***
mdatbio$year_label <- paste0(mdatbio$year, " - F1: ", round(mdatbio$f1, 2), " AUC: ", round(mdatbio$auc, 2), " AP: ", round(mdatbio$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

p5 <- ggplot(mdatbio, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision", color="Oceanographic Variables") +
  theme_tufte(11) +
  scale_color_manual(values = viridis(6, option = "D")) +
  geom_line() +
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.28, .40),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.title.align = 0.5,
        #legend.text = element_text(size=8),
        #legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
p5






plot_grid(p4, p5, p1, p2, p3, ncol=2, labels = c("(A)", "(B)", "(C)", "(D)", "(E)"), label_x = .89, label_y = .98)

ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1.pdf", width = 10, height = 8)
ggsave("~/Projects/predicting-illegal-fishing/figures/FigureS1.png", width = 10, height = 8)
