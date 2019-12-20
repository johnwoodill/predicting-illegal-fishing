library(tidyverse)
library(ggthemes)
library(feather)
library(viridis)
library(lubridate)
library(stringr)

# 8-day Data
dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

# Fishing Effort predictions
fe <- read_feather('~/Projects/predicting-illegal-fishing/data/predicted_effort_data.feather')

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")


dat$lat_lon <- paste0(dat$lat1, "_", dat$lon1)

dat2 <- dat %>% group_by(date, lat_lon) %>% 
  summarise(fh = sum(fishing_hours))
nrow(dat)


# ------------------------------------------------------------------------------------
# Figure 1. Number of illegal events per month and year
mapdat <- dat
mapdat$year <- year(mapdat$date)
mapdat$month <- month(mapdat$date)
mapdat$year_month <- paste0(mapdat$month, "-", mapdat$year)
mapdat <- filter(mapdat, flag %in% c("ARG", "CHN"))
mapdat$illegal <- ifelse(mapdat$eez == TRUE, ifelse(mapdat$flag != "ARG", ifelse(mapdat$fishing_hours > 0, 1, 0), 0), 0)
mapdat$lat_lon <- paste0(mapdat$lat1, "_", mapdat$lon1)

mapdat2 <- mapdat %>% 
  group_by(lat_lon) %>% 
  summarise(sum_illegal = sum(illegal),
            lat = mean(lat1),
            lon = mean(lon1))

mapdat2

# Correct 4/24/2019
bat <- getNOAA.bathy(-68, -51, -51, -39, res = 1, keep = TRUE)
bat2 <- getNOAA.bathy(-77, -22, -58, -23, res = 1, keep = TRUE)


LON1 = -68
LON2 = -51
LAT1 = -51
LAT2 = -39


loc = c(-58, -22)
map1 <- ggmap(get_map(loc, zoom = 3, maptype='toner-background', color='bw', source='stamen')) + 
  theme_nothing() + 
  geom_segment(x=-68, xend=-68, y=-39, yend=-51, color='red') +
  geom_segment(x=-68, xend=-51, y=-51, yend=-51, color='red') +
  geom_segment(x=-51, xend=-51, y=-51, yend=-39, color='red') +
  geom_segment(x=-68, xend=-51, y=-39, yend=-39, color='red') +
  labs(x=NULL, y=NULL) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  scale_y_continuous(expand=c(0,0)) +
  scale_x_continuous(expand=c(0,0)) +
  NULL
map1


ggplot(NULL) + 
  geom_tile(data = filter(mapdat2, sum_illegal > 0), aes(lon, lat, fill=sum_illegal), size=50) +
  scale_fill_distiller(palette="Spectral", na.value="white") +
  NULL




map2 <- 
  autoplot(bat, geom = c("raster", "contour")) +
  geom_raster(aes(fill=z)) +
  geom_contour(aes(z = z), colour = "white", alpha = 0.05) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 30, 40, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", 
                                  "grey50", "grey70", "grey85")) +
  labs(x=NULL, y=NULL, color="Illegal \n Activity") +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  geom_point(data = filter(mapdat2, sum_illegal > 1), aes(lon, lat, color=sum_illegal), shape=15) +
  annotate("text", x=-64.25, y = -39.25, label="Patagonia Shelf, Argentina", size = 4, color='black', fontface=2) +
  annotate("text", x=-65.25, y = -39.75, label="Total Illegal Activity", size = 4, color='black', fontface=2) +
  annotate("text", x=-66.25, y = -40.25, label="2012-2016", size = 4, color='black', fontface=2) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.position = c(.93, 0.2),
        legend.margin=margin(l = 0, unit='cm'),
        legend.text = element_text(size=10),
        legend.title = element_text(size=12),
        legend.key = element_rect(fill = "transparent", colour = "transparent"),
        legend.background = element_rect(fill = "transparent", colour = "transparent"),
        panel.grid = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  scale_color_gradientn(colours=brewer.pal(9, "OrRd"), limits=c(0, 200)) +
  
  # Legend up top
  guides(fill = FALSE,
         color = guide_colorbar(title.hjust = unit(1.1, 'cm'),
                                title.position = "top",
                                frame.colour = "black",
                                barwidth = .5,
                                barheight = 7,
                                label.position = 'left')) +
  # scale_y_continuous(expand=c(0,0)) +
  # scale_x_continuous(expand=c(0,0)) +
  NULL

map2



ggdraw() + draw_plot(map2, 0, 0, height = 1, width = 1) +
  draw_plot(map1, .025, .024, height = .26, width = .25)

ggsave("~/Projects/predicting-illegal-fishing/figures/Figure1.pdf", width=5, height=5)





# Figure 2. Number of illegal events per month and year
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
  scale_fill_grey() +
  theme_tufte(12) +
  labs(x=NULL, y="Count of Illegal Chinese \n Vessels Fishing") +
  theme(legend.position = c(.95, .825),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.text = element_text(size=8),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  guides(fill = guide_legend(reverse=TRUE)) +
  NULL


ggsave("~/Projects/predicting-illegal-fishing/figures/Figure2.pdf", width=8, height=4)
ggsave("~/Projects/predicting-illegal-fishing/figures/Figure2.png", width=8, height=4)




# Figure 3. Presion-recall plot
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
mdat <- read_feather('~/Projects/predicting-illegal-fishing/data/illegal_cross_val_dat.feather')

mdat$year_label <- paste0(mdat$year, " - F1: ", round(mdat$f1, 2), " AUC: ", round(mdat$auc, 2), " AP: ", round(mdat$ap, 2))

# Custom color palette
cbp1 <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
          "#0072B2", "#D55E00", "#CC79A7")

ggplot(mdat, aes(recall, prec, color=factor(year_label))) + 
  labs(x="Recall", y="Precision") +
  theme_tufte(12) +
  scale_color_manual(values = viridis(6, option = "D")) +
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

ggsave("~/Projects/predicting-illegal-fishing/figures/Figure3.pdf", width = 6, height = 4)
ggsave("~/Projects/predicting-illegal-fishing/figures/Figure3.png", width = 6, height = 4)




# Figure 4. Feature Importance
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# To plot radar plot below
coord_radar <- function (theta = "x", start = 0, direction = 1) {
  theta <- match.arg(theta, c("x", "y"))
  r <- if (theta == "x") "y" else "x"
  ggproto("CordRadar", CoordPolar, theta = theta, r = r, start = start, 
          direction = sign(direction),
          is_linear = function(coord) TRUE)
}

fea <- read_feather('~/Projects/predicting-illegal-fishing/data/feature_importance_rf_illegal.feather')

fea_dat <- data.frame("variable" = c("distance_to_eez_km", "eez", "coast_dist_km", "lon1", "port_dist_km", "lat1",
                                   "sst", "chlor_a", "fishing_hours", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                   "Aug", "Sep", "Oct", "Nov", "Dec", "seascape_1.0", "seascape_2.0", "seascape_3.0", "seascape_4.0"
                                   , "seascape_5.0", "seascape_6.0", "seascape_7.0", "seascape_8.0", "seascape_9.0"
                                   , "seascape_10.0", "seascape_11.0", "seascape_12.0", "seascape_13.0", "seascape_14.0"
                                   , "seascape_15.0", "seascape_16.0", "seascape_17.0", "seascape_18.0", "seascape_19.0"
                                   , "seascape_20.0", "seascape_21.0", "seascape_22.0", "seascape_23.0", "seascape_24.0"
                                   , "seascape_25.0", "seascape_26.0", "seascape_27.0", "seascape_28.0", "seascape_29.0"
                                   , "seascape_30.0", "seascape_31.0", "seascape_32.0", "seascape_33.0", "seascape_34.0"),
                      "labels" = c("Distance to EEZ", "EEZ", "Distance to Coast", "Longitude", "Distance to Port", "Latitude",
                                   "Sea Surface Temp.", "Chlorophyll", "Fishing Hours", "January", "February", "March", "April",
                                   "May", "June", "July", "August", "September", "October", "November", "December", "Seascape 1", "Seascape 2", "Seascape 3", "Seascape 4"
                                   , "Seascape 5", "Seascape 6", "Seascape 7", "Seascape 8", "Seascape 9"
                                   , "Seascape 10", "Seascape 11", "Seascape 12", "Seascape 13", "Seascape 14"
                                   , "Seascape 15", "Seascape 16", "Seascape 17", "Seascape 18", "Seascape 19"
                                   , "Seascape 20", "Seascape 21", "Seascape 22", "Seascape 23", "Seascape 24"
                                   , "Seascape 25", "Seascape 26", "Seascape 27", "Seascape 28", "Seascape 29"
                                   , "Seascape 30", "Seascape 31", "Seascape 32", "Seascape 33", "Seascape 34"))

fea <- left_join(fea, fea_dat, by = "variable")
fea <- dplyr::select(fea, labels, importance, year)

# Get top five from each  year
fea <- fea %>% 
  group_by(year) %>% 
  arrange(-importance) %>% 
  do(head(., n = 5)) %>% 
  ungroup()


fead <- fead %>% group_by(year) %>% 
  arrange(labels, year) %>% 
  ungroup()
fead

ggplot(fead, aes(x=labels, y=importance, color=factor(year), group=factor(year))) + 
  geom_polygon(fill=NA) +
  labs(y=NULL, x=NULL, color=NULL) + 
  geom_point(size = 1) +
  theme_bw() +
  guides(color = guide_legend(keywidth = 1, keyheight = 1,
                       override.aes = list(size = .5, shape = NA))) +
  theme(panel.grid.major = element_line(colour = "grey"), 
        legend.position = c(.9, 0.15),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  coord_radar() +
  annotate("text", x=.5, y=0.35, label = "0.35", color='darkgrey',vjust=-.45) + 
  annotate("text", x=.5, y=0.30, label = "0.30", color='darkgrey',vjust=-.45) + 
  annotate("text", x=.5, y=0.25, label = "0.25", color='darkgrey',vjust=-.45) + 
  annotate("text", x=.5, y=0.20, label = "0.20", color='darkgrey',vjust=-.45) + 
  annotate("text", x=.5, y=0.15, label = "0.15", color='darkgrey',vjust=-.45) + 
  NULL
  

  

ggsave("~/Projects/predicting-illegal-fishing/figures/Figure4.pdf", width = 5, height = 4.5)
ggsave("~/Projects/predicting-illegal-fishing/figures/Figure4.png", width = 5, height = 4.5)





# Figure 5. Movement from legal to illegal seascape
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
dat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

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

dat4 <- select(dat3, date, mmsi, sea_change, ill_change, seascape_class, seascape_lag, seascape_lead, nominal)
dat4 <- drop_na(dat4)

p1 <- ggplot(dat4, aes(reorder(seascape_lag, seascape_lag, function(x) length(x)))) + 
  labs(x=NULL, y="Number of Vessels Moving \n from Legal to Illegal") +
  theme_tufte(12) +
  geom_bar() + 
  coord_flip() +
  ylim(0, 85) +
  # annotate("text", x=5.3, y = 90, label="(a)", size = 4, color='black', fontface=2) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  geom_text(stat='count', aes(label=..count..), hjust=-.20, size = 3)
p1
ggsave("~/Projects/predicting-illegal-fishing/figures/Figure5.pdf", width=6, height=3)
ggsave("~/Projects/predicting-illegal-fishing/figures/Figure5.png", width=6, height=3)



# Figure 6. Legal to Illegal Seascape Map
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
dat = read_feather("~/Projects/predicting-illegal-fishing/data/illegal_seascape_pred.feather")

dat = read_feather("~/Projects/predicting-illegal-fishing/data/illegal_seascape_data_model.feather")

idat = read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

idat$illegal <- ifelse(idat$eez == TRUE, ifelse(idat$flag == "CHN", ifelse(idat$fishing_hours > 0, 1, 0), 0), 0)


# Seascapes (2016-02-26)
sdat = filter(dat, !is.na(seascape_class) & date == '2016-02-26' & (seascape_class %in% c(14)))
sdat$seascape_class <- "Temperate Blooms Upwelling"

ildat = filter(idat, date == '2016-02-26')
ildat$illegal <- ifelse(ildat$illegal == 1, "Illegal", "Legal")
date_ = "2016-02-26"



p1 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  geom_point(data = NULL, aes(x=-67.65, y = -40.05, shape=factor("temp_bloom")), size = 2, shape = 19, color="cornflowerblue") +
  geom_point(data=sdat, aes(x=lon, y=lat, color=factor(seascape_class)), color="cornflowerblue", size = 0.5) +
  geom_point(data=ildat, aes(lon1, lat1, color=factor(illegal)), size=0.5) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  labs(x=NULL, y=NULL) +
  annotate("text", x=-66.25, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  annotate("text", x=-65.1, y = -39.65, label="# Illegal Vessels = 76", size = 4, color='black', fontface=2) +
  annotate("text", x=-63.65, y = -40.05, label="Temperate Blooms Upwelling", size = 4, fontface=2) +
  annotate("text", x=-51.5, y = -39.25, label="(A)", size = 4, color='black', fontface=2) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = c(.901, 0.07),
        legend.margin=margin(t = -.15, r = .25, b = .05, l = .05, unit='cm'),
        panel.grid = element_blank(),
        legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'),
        legend.direction = 'vertical',
        legend.title = element_blank(),
        legend.key=element_blank()) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  scale_color_manual(values = c("Legal" = "black", "Illegal" = "red", "Temperate Blooms Upwelling" = "cornflowerblue")) +
  NULL
p1

# ggsave("~/Projects/predicting-illegal-fishing/figures/Figure6a.pdf", width=5, height = 5)


# Seascapes (2016-01-25)
sdat = filter(dat, !is.na(seascape_class) & date == '2016-01-25' & (seascape_class %in% c(14)))
sdat$seascape_class <- "Temperate Blooms Upwelling"
ildat = filter(idat, date == '2016-01-25')
ildat$illegal <- ifelse(ildat$illegal == 1, "Illegal", "Legal")
date_ = "2016-01-25"

p2 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data=sdat, aes(x=lon, y=lat, color=factor(seascape_class)), color="cornflowerblue", size = 0.5) +
  geom_point(data=ildat, aes(lon1, lat1, color=factor(illegal)), size=0.5) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  labs(x=NULL, y=NULL) +
  geom_point(data = NULL, aes(x=-67.65, y = -40.05, shape=factor("temp_bloom")), size = 2, shape = 19, color="cornflowerblue") +
  annotate("text", x=-66.25, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  annotate("text", x=-65.25, y = -39.65, label="# Illegal Vessels = 3", size = 4, color='black', fontface=2) +
  annotate("text", x=-63.65, y = -40.05, label="Temperate Blooms Upwelling", size = 4, fontface=2) +
  annotate("text", x=-51.5, y = -39.25, label="(B)", size = 4, color='black', fontface=2) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = c(.901, 0.07),
        legend.margin=margin(t = -.15, r = .25, b = .05, l = .05, unit='cm'),
        panel.grid = element_blank(),
        legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'),
        legend.direction = 'vertical',
        legend.title = element_blank(),
        legend.key=element_blank()) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  scale_color_manual(values = c("Legal" = "black", "Illegal" = "red", "Temperate Blooms Upwelling" = "cornflowerblue")) +
  NULL
p2

# ggsave("~/Projects/predicting-illegal-fishing/figures/Figure6b.pdf", width=5, height = 5)

plot_grid(p1, p2, ncol=2)


ggsave("~/Projects/predicting-illegal-fishing/figures/Figure6.pdf", width=10, height = 5)
NULL



# Figure *** Fishing effort explained
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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





