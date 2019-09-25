library(tidyverse)
library(ggthemes)
library(feather)
library(viridis)
library(lubridate)
library(stringr)

# Full data
dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Model results
mdat <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_cross_val_dat.feather')

# Feature importance
fea <- read_feather('~/Projects/predicting-illegal-fishing/data/feature_importance_rf_illegal.feather')

# Fishing Effort predictions
fe <- read_feather('~/Projects/predicting-illegal-fishing/data/predicted_effort_data.feather')




# ------------------------------------------------------------------------------------
# Figure ***
dat$year <- year(dat$date)
month <- month(dat$date)
dat$year_month <- paste0(dat$month, "-", dat$year)
dat <- filter(dat, fishing_hours > 0)
dat <- filter(dat, flag %in% c("ARG", "CHN"))
dat$illegal <- ifelse(dat$eez == TRUE, ifelse(dat$flag != "ARG", 1, 0), 0)

pdat <- dat %>% 
  group_by(year_month) %>% 
  summarise(year = mean(year), 
            month = mean(month),
            total_illegal = sum(illegal))

pdat$month_name <- month.name[pdat$month]
pdat <- arrange(pdat, month_name)
pdat$month_name <- factor(pdat$month_name, levels = month.name)

ggplot(pdat, aes(year_month, total_illegal, fill=factor(month_name))) + 
  theme_tufte(12) +
  labs(x=NULL, y="Count of Illegal Chinese Activity") +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d() +
  theme(legend.title = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  coord_flip() +
  NULL








# ------------------------------------------------------------------------------------
# Figure *** Feature Importance
fea <- filter(fea, importance > 0)
fea$variable <- c("Distance to EEZ",
                  "EEZ",
                  "Distance to Coast",
                  "Longitude",
                  "Distance to Port",
                  "Latitude",
                  "Sea Surface Temp.",
                  "Chlorophyll",
                  "Ocean Depth",
                  "Sea Surface Gradient",
                  "August",
                  "September",
                  "Seascape 14",
                  "Seascape 7",
                  "February",
                  "December",
                  "May",
                  "March",
                  "June",
                  "July",
                  "January",
                  "April",
                  "November",
                  "October",
                  "Seascape 21",
                  "Seascape 15",
                  "Seascape 27",
                  "Seascape 17",
                  "Seascape 3",
                  "Seascape 12")

ggplot(fea, aes(reorder(variable, importance), importance)) + 
  geom_bar(stat='identity') +
  theme_tufte(12) +
  labs(x=NULL, y="Feature Importance") +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  coord_flip() +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/feature_importance.pdf", width = 6, height = 5)








# ------------------------------------------------------------------------------------
# Figure ***
mdat$year_label <- paste0(mdat$year, " - F1: ", round(mdat$f1, 2), " AUC: ", round(mdat$auc, 2), " AP: ", round(mdat$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mdat, aes(recall, prec, color=factor(year_label))) + 
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

ggsave("~/Projects/predicting-illegal-fishing/figures/precision_recall.pdf", width = 6, height = 4)







# ------------------------------------------------------------------------------------
# Figure *** Fishing effort explained

fe <- gather(fe, key=label, value = value, -year)
fe <- filter(fe, label != "total_fishing")

fe$label <- ifelse(fe$label == "tpr_fishing", "Explained Illegal Fishing Effort", "Unexplained Illegal Fishing Effort")
fe$label <- factor(fe$label, levels = c("Unexplained Illegal Fishing Effort", "Explained Illegal Fishing Effort"))

ggplot(fe, aes(x=year, y=value, fill=factor(label))) + 
  geom_bar(stat='identity') +
  labs(x=NULL, y="Total Illegal Fishing Hours") +
  theme_tufte(12) +
  scale_fill_manual(values = c("#22A884FF", "#2A788EFF")) +
  theme(legend.position = c(.20, .90),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL

ggsave("~/Projects/predicting-illegal-fishing/figures/explained_fishing_effort.pdf", width=6, height=4)





