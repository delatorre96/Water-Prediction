CREATE TABLE `df_pixeles_cercanos` (
  `location_id_copernicus` int64,
  `location_id_embalses` float64,
  `location_id_aemet` float64,
  `location_id_rios_canales` float64
);

CREATE TABLE `df_copernicus` (
  `total_precipitation` float64,
  `skin_temperature` float64,
  `evaporation` float64,
  `runoff` float64,
  `snowfall` float64,
  `soil_water_l1` float64,
  `soil_water_l2` float64,
  `soil_water_l3` float64,
  `soil_water_l4` float64,
  `high_vegetation_cover` float64,
  `low_vegetation_cover` float64,
  `type_high_vegetation` float64,
  `type_low_vegetation` float64,
  `location_id` int64,
  `date_id` int64
);

CREATE TABLE `df_embalses` (
  `quantity_hm3` float64,
  `location_id` int64,
  `date_id` int64
);

CREATE TABLE `df_embalses_info` (
  `id_station` int64,
  `Embalse` varchar(255),
  `CodROEA` int64,
  `CodPresaprincipal` varchar(255),
  `PresaPrincipal` varchar(255),
  `VolUtil_hm3` float64,
  `CodMunic` int64,
  `Municipio` varchar(255),
  `CodProv` int64,
  `Provincia` varchar(255),
  `CodSE` int64,
  `SistemadeExplotación` varchar(255),
  `Cauce` varchar(255),
  `CodMasaSuperfPHJ22` varchar(255),
  `MasaSuperficialPHJ22` varchar(255),
  `location_id` int64
);

CREATE TABLE `df_rios_canales` (
  `quantity_hm3` float64,
  `location_id` int64,
  `date_id` int64
);

CREATE TABLE `df_rios_canales_info` (
  `id_station` int64,
  `EstacióndeAforo` varchar(255),
  `CodROEA` int64,
  `Tipo` varchar(255),
  `CodMunic` int64,
  `Municipio` varchar(255),
  `CodSE` float64,
  `SistemadeExplotación` varchar(255),
  `Altitud` varchar(255),
  `location_id` int64
);

CREATE TABLE `df_aemet` (
  `altitud` int64,
  `tmed` float64,
  `prec` float64,
  `tmin` float64,
  `tmax` float64,
  `dir` float64,
  `velmedia` float64,
  `racha` float64,
  `horaracha` varchar(255),
  `hrMin` float64,
  `presMax` float64,
  `presMin` float64,
  `sol` float64,
  `location_id` int64,
  `date_id` int64
);

CREATE TABLE `df_aemet_info` (
  `provincia` object,
  `altitud` int64,
  `indicativo` object,
  `nombre` object,
  `indsinop` float64,
  `location_id` int64
);

CREATE TABLE `df_date` (
  `date` timestamp,
  `date_id` int64
);

CREATE TABLE `locations_id` (
  `latitude` float64,
  `longitude` float64,
  `location_id` int64,
  `Type` object
);

ALTER TABLE `df_date` ADD FOREIGN KEY (`date_id`) REFERENCES `df_copernicus` (`date_id`);

ALTER TABLE `df_date` ADD FOREIGN KEY (`date_id`) REFERENCES `df_aemet` (`date_id`);

ALTER TABLE `df_date` ADD FOREIGN KEY (`date_id`) REFERENCES `df_rios_canales` (`date_id`);

ALTER TABLE `df_date` ADD FOREIGN KEY (`date_id`) REFERENCES `df_embalses` (`date_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_copernicus`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_rios_canales`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_embalses`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_aemet`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_copernicus`) REFERENCES `df_copernicus` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_rios_canales`) REFERENCES `df_rios_canales` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_embalses`) REFERENCES `df_embalses` (`location_id`);

ALTER TABLE `df_pixeles_cercanos` ADD FOREIGN KEY (`location_id_aemet`) REFERENCES `df_aemet` (`location_id`);

ALTER TABLE `df_rios_canales_info` ADD FOREIGN KEY (`location_id`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_embalses_info` ADD FOREIGN KEY (`location_id`) REFERENCES `locations_id` (`location_id`);

ALTER TABLE `df_aemet_info` ADD FOREIGN KEY (`location_id`) REFERENCES `locations_id` (`location_id`);
