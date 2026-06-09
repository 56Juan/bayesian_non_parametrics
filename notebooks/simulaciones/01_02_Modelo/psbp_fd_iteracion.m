%run_pipeline.m
% Script principal para ejecutar experimento desde MATLAB
% Alineado con parámetros de notebook Python

clear; clc; close all;

%PARÁMETROS DEL EXPERIMENTO (Mismos que el nootbook)           

TT       = 1;
SEED     = 4123;
BASENAME = "modelo_unificado";
paths = config_paths(BASENAME, TT, SEED);

rng(SEED);  % Reproducibilidad
fprintf("RNG seed establecida: %d\n\n", SEED);
