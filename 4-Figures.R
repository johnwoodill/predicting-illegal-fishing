library(tidyverse)
library(ggthemes)
library(feather)
library(viridis)
library(lubridate)
library(stringr)

# Full data
# dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Model results
mdat <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_cross_val_dat.feather')

# Feature importance
fea <- read_feather('~/Projects/predicting-illegal-fishing/data/feature_importance_rf_illegal.feather')

# Fishing Effort predictions
fe <- read_feather('~/Projects/predicting-illegal-fishing/data/predicted_effort_data.feather')

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")



# ------------------------------------------------------------------------------------
# Figure ***
dat$year <- year(dat$date)
dat$month <- month(dat$date)
dat$year_month <- paste0(dat$month, "-", dat$year)
dat <- filter(dat, flag %in% c("ARG", "CHN"))
dat$illegal <- ifelse(dat$eez == TRUE, ifelse(dat$flag != "ARG", ifelse(dat$fishing_hours > 0, 1, 0), 0), 0)

pdat <- dat %>% 
  group_by(year_month) %>% 
  summarise(year = mean(year), 
            month = mean(month),
            total_illegal = sum(illegal))

pdat$month_name <- month.abb[pdat$month]
pdat <- arrange(pdat, month_name)
pdat$month_name <- factor(pdat$month_name, levels = month.abb)

ggplot(pdat, aes(month_name, total_illegal, fill=factor(year))) + 
  geom_bar(stat='identity', position = position_stack(reverse = TRUE), width = 0.75) +
  scale_fill_viridis_d(option = "D") +
  theme_tufte(12) +
  labs(x=NULL, y="Count of Illegal Chinese Activity") +
  theme(legend.position = c(.95, .825),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL


ggsave("~/Projects/predicting-illegal-fishing/figures/sum_illegal_activity.pdf", width=8, height=4)







# ------------------------------------------------------------------------------------
# Figure *** Feature Importance
fea <- filter(fea, importance > 0)
fea$variable

fea$variable <- c("Distance to EEZ",
                  "EEZ",
                  "Distance to Coast",
                  "Longitude",
                  "Latitude",
                  "Distance to Port",
                  "Sea Surface Temp.",
                  "Chlorophyll",
                  "Fishing Hours",
                  # "Ocean Depth",
                  # "Sea Surface Gradient",
                  "February",
                  "Seascape 14",
                  "Seascape 12",
                  "March",
                  "September",
                  "Seascape 7",
                  "April",
                  "May",
                  "January",
                  "December",
                  "August",
                  "July",
                  "November",
                  "October", 
                  "June",
                  "Seascape 21",
                  "Seascape 27",
                  "Seascape 17",
                  "Seascape 19",
                  "Seascape 15",
                  "Seascape 25",
                  "Seascape 2")

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
  theme(legend.background = element_rect(colour = 'grey', fill = 'white', linetype='solid'),
        legend.position = c(.225, .25),
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



# ---------------------------------------------------------
# Not illegal to illegal seascape 

dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

seascape_labels <- data.frame(seascape_class = (seq(1, 33)),
                              nominal =(c("NORTH ATLANTIC SPRING, ACC TRANSITION",
                                          "SUBPOLAR TRANSITION",
                                          "TROPICAL SUBTROPICAL TRANSITION",
                                          "WESTERN WARM POOL SUBTROPICAL",
                                          "SUBTROPICAL GYRE TRANSITION",
                                          "ACC, NUTRIENT STRESS",
                                          "TEMPERATE TRANSITION",
                                          "INDOPACIFIC SUBTROPICAL GYRE",
                                          "EQUATORIAL TRANSITION",
                                          "HIGHLY OLIGOTROPHIC SUBTROPICAL GYRE",
                                          "TROPICAL/SUBTROPICAL UPWELLING",
                                          "SUBPOLAR",
                                          "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                          "TEMPERATE BLOOMS UPWELLING",
                                          "TROPICAL SEAS",
                                          "MEDITTERANEAN RED SEA",
                                          "SUBTROPICAL TRANSITION \n LOW NUTRIENT STRESS",
                                          "MEDITTERANEAN RED SEA",
                                          "ARTIC/ SUBPOLAR SHELVES",
                                          "SUBTROPICAL, FRESH INFLUENCED COASTAL",
                                          "WARM, BLOOMS, HIGH NUTS",
                                          "ARCTIC LATE SUMMER",
                                          "FRESHWATER INFLUENCED POLAR SHELVES",
                                          "ANTARCTIC SHELVES",
                                          "CE PACK",
                                          "ANTARCTIC ICE EDGE",
                                          "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                          "ARCTIC ICE EDGE","ANTARCTIC",
                                          "ICE EDGE  BLOOM",
                                          "1-30% ICE PRESENT",
                                          "30-80% MARGINAL ICE","PACK ICE")))

dat$year <- year(dat$date)
dat$month <- month(dat$date)
dat$year_month <- paste0(dat$month, "-", dat$year)
dat <- filter(dat, flag %in% c("ARG", "CHN"))
dat$illegal <- ifelse(dat$eez == TRUE, ifelse(dat$flag != "ARG", ifelse(dat$fishing_hours > 0, 1, 0), 0), 0)

dat <- left_join(dat, seascape_labels, by='seascape_class')

dat2 <- dat %>% 
  group_by(mmsi) %>% 
  arrange(date) %>% 
  mutate(seascape_lag = lag(nominal),
         seascape_lead = lead(nominal),
         illegal_lag = lag(illegal),
         illegal_lead= lead(illegal))

dat2$sea_change <- ifelse(dat2$nominal == dat2$seascape_lag, 0, 1)
dat2$ill_change <- ifelse(dat2$illegal == 1, ifelse(dat2$illegal_lag == 0, 1, 0), 0)

dat3 <- filter(dat2, sea_change == 1 & ill_change == 1)
dat3

dat4 <- select(dat3, date, mmsi, sea_change, ill_change, seascape_class, seascape_lag, seascape_lead, nominal)
dat4 <- drop_na(dat4)

ggplot(dat4, aes(reorder(seascape_lag, seascape_lag, function(x) length(x)))) + 
  labs(x=NULL, y="Number of Illegal Vessels") +
  theme_tufte(12) +
  geom_bar() + 
  coord_flip() +
  theme(panel.border = element_rect(colour = "grey", fill=NA, size=1)) +
  geom_text(stat='count', aes(label=..count..), hjust=-.15)

ggsave("~/Projects/predicting-illegal-fishing/figures/illegal_seascapes_change.pdf", width=8, heigh=4)





# --------------------------------------------------------------------------------------------
# Mapping figures

dat <- read_feather('~/Projects/Seascape-and-fishing-effort/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')
dat$year <- year(dat$date)
dat$month <- month(dat$date)
dat$year_month <- paste0(dat$month, "-", dat$year)
dat <- filter(dat, flag %in% c("ARG", "CHN"))
dat$illegal <- ifelse(dat$eez == TRUE, ifelse(dat$flag != "ARG", ifelse(dat$fishing_hours > 0, 1, 0), 0), 0)



dat2 <- filter(dat, illegal == TRUE)
dat2$year_month

dat3 <- dat2 %>% 
  group_by(year_month) %>% 
  summarise(sum_fh = sum(fishing_hours))

arrange(dat3, -sum_fh)
arrange(dat3, sum_fh)

ggplot(dat3, aes(x=year_month, y=sum_fh)) + geom_bar(stat='identity')




# Check EEZ Map

eez <- read_csv("~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_EEZ.csv")
eez <- filter(eez, lon >= -68 & lon <= -51 & lat >= -51 & lat <= -39)
eez <- filter(eez, order <= 25190)

bat <- getNOAA.bathy(-68, -51, -51, -39, res = 1, keep = TRUE)


dat <- read_feather('~/Projects/Seascape-and-fishing-effort/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')






autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE) +
  geom_raster(aes(fill=z)) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01) +
  
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data = dat2, aes(x=lon1, y=lat1, color=distance_to_eez_km)) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  # annotate("text", x=-54.5, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  # theme(axis.title.x=element_blank(),
  #       axis.text.x=element_blank(),
  #       axis.ticks.x=element_blank(),
  #       axis.title.y=element_blank(),
  #       axis.text.y=element_blank(),
  #       axis.ticks.y=element_blank(),
  #       legend.direction = 'vertical',
  #       legend.justification = 'center',
  #       legend.position = "none",
  #       legend.margin=margin(l = 0, unit='cm'),
#       legend.text = element_text(size=10),
#       legend.title = element_text(size=12),
#       panel.grid = element_blank()) +

# Legend up top
annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  # scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) +
  NULL








