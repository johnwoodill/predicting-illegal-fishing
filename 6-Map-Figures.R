library(feather)
library(tidyverse)
library(cowplot)

dat = read_feather("~/Projects/predicting-illegal-fishing/data/illegal_seascape_pred.feather")

dat = read_feather("~/Projects/predicting-illegal-fishing/data/illegal_seascape_data_model.feather")

idat = read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

idat$illegal <- ifelse(idat$eez == TRUE, ifelse(idat$flag == "CHN", ifelse(idat$fishing_hours > 0, 1, 0), 0), 0)

seascape_labels <- data.frame(seascape = as.factor(seq(1, 33)),
                              nominal = as.factor(c("NORTH ATLANTIC SPRING, ACC TRANSITION",
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
                                                    "ICE PACK",
                                                    "ANTARCTIC ICE EDGE",
                                                    "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                                    "ARCTIC ICE EDGE","ANTARCTIC",
                                                    "ICE EDGE  BLOOM",
                                                    "1-30% ICE PRESENT",
                                                    "30-80% MARGINAL ICE","PACK ICE")))

dat$seascape_label <- left_join(dat, seascape_labels, by=seascape)




# Illegal Maps

ggplot(ndat, aes(lon1, lat1, color=sst)) + geom_point()


date_ = dat$date[0]

# Sea surface temperature (2016-02-26)
sdat = filter(dat, !is.na(sst) | date == '2016-02-26')

p1 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data=sdat, aes(x=lon, y=lat, color=sst), size = 0.5) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  labs(x=NULL, y=NULL) +
  guides(color = guide_colorbar(label.hjust = unit(0, 'cm'),
                                frame.colour = "black",
                                barwidth = .5,
                                barheight = 12)) +
  # Legend up top
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  # scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) +
  NULL
p1

# Sea surface temperature (2016-02-26)
sdat = filter(dat, !is.na(sst) | date == '2016-01-25')

p2 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data=sdat, aes(x=lon, y=lat, color=sst), size = 0.5) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  labs(x=NULL, y=NULL) +
  guides(color = guide_colorbar(label.hjust = unit(0, 'cm'),
                                frame.colour = "black",
                                barwidth = .5,
                                barheight = 12)) +
  # Legend up top
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  # scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) +
  NULL
p2

# Seascapes (2016-02-26)
sdat = filter(dat, !is.na(seascape_class) & date == '2016-02-26' & (seascape_class %in% c(12, 14)))
ildat = filter(idat, date == '2016-02-26')
date_ = "2016-02-26"

p3 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data=sdat, aes(x=lon, y=lat, color=factor(seascape_class)), size = 0.5) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  geom_point(data=ildat, aes(lon1, lat1, color=factor(illegal)), size=0.5) +
  labs(x=NULL, y=NULL, title='illegal') +
  annotate("text", x=-65.7, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  annotate("text", x=-54.6, y = -39.25, label="# Illegal Vessels = 76", size = 4, color='black', fontface=2) +
  # guides(color = guide_colorbar(label.hjust = unit(0, 'cm'),
  #                               frame.colour = "black",
  #                               barwidth = .5,
  #                               barheight = 12)) +
  # Legend up top
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  scale_color_manual(values = c("0" = "black", "1" = "red", "12" = "darkblue", "14" = "orange")) +
  NULL
p3


# Seascapes (2016-01-25)
sdat = filter(dat, !is.na(seascape_class) & date == '2016-01-25' & (seascape_class %in% c(12, 14)))
ildat = filter(idat, date == '2016-01-25')
date_ = "2016-01-25"

p4 <- autoplot.bathy(bat, geom = c("contour", "raster"), coast=TRUE, show.legend = FALSE) +
  geom_raster(aes(fill=z), show.legend = FALSE) +
  geom_contour(aes(z = z), color = "white", alpha = 0.01, show.legend = FALSE) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 39, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", "grey50", "grey80")) +
  
  geom_point(data=sdat, aes(x=lon, y=lat, color=factor(seascape_class)), size = 0.25) +
  geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
  geom_point(data=ildat, aes(lon1, lat1, color=factor(illegal)), size=.5) +
  labs(x=NULL, y=NULL, title = "legal") +
  annotate("text", x=-65.7, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  annotate("text", x=-54.6, y = -39.25, label="# Illegal Vessels = 3", size = 4, color='black', fontface=2) +
  # guides(color = guide_colorbar(label.hjust = unit(0, 'cm'),
  #                               frame.colour = "black",
  #                               barwidth = .5,
  #                               barheight = 12)) +
  # Legend up top
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) +  # Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) +  # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) +    # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) +    # Top
  scale_color_manual(values = c("0" = "black", "1" = "red", "12" = "darkblue", "14" = "orange")) +
  NULL
p4

plot_grid(p3, p4, ncol=2)


ggsave("~/Projects/predicting-illegal-fishing/figures/SEA_map.pdf", width=12, height = 6)




