function mpc = case39_DCOPF
%CASE39_DCOPF

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	1	97.6	44.2	0	0	2	1	0	345	1	1.06	0.94;
	2	1	0	0	0	0	2	1	0	345	1	1.06	0.94;
	3	1	322	2.4	0	0	2	1	0	345	1	1.06	0.94;
	4	1	500	184	0	0	1	1	0	345	1	1.06	0.94;
	5	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	6	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	7	1	233.8	84	0	0	1	1	0	345	1	1.06	0.94;
	8	1	522	176.6	0	0	1	1	0	345	1	1.06	0.94;
	9	1	6.5	-66.6	0	0	1	1	0	345	1	1.06	0.94;
	10	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	11	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	12	1	8.53	88	0	0	1	1	0	345	1	1.06	0.94;
	13	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	14	1	0	0	0	0	1	1	0	345	1	1.06	0.94;
	15	1	320	153	0	0	3	1	0	345	1	1.06	0.94;
	16	1	329	32.3	0	0	3	1	0	345	1	1.06	0.94;
	17	1	0	0	0	0	2	1	0	345	1	1.06	0.94;
	18	1	158	30	0	0	2	1	0	345	1	1.06	0.94;
	19	1	0	0	0	0	3	1	0	345	1	1.06	0.94;
	20	1	680	103	0	0	3	1	0	345	1	1.06	0.94;
	21	1	274	115	0	0	3	1	0	345	1	1.06	0.94;
	22	1	0	0	0	0	3	1	0	345	1	1.06	0.94;
	23	1	247.5	84.6	0	0	3	1	0	345	1	1.06	0.94;
	24	1	308.6	-92.2	0	0	3	1	0	345	1	1.06	0.94;
	25	1	224	47.2	0	0	2	1	0	345	1	1.06	0.94;
	26	1	139	17	0	0	2	1	0	345	1	1.06	0.94;
	27	1	281	75.5	0	0	2	1	0	345	1	1.06	0.94;
	28	1	206	27.6	0	0	3	1	0	345	1	1.06	0.94;
	29	1	283.5	26.9	0	0	3	1	0	345	1	1.06	0.94;
	30	3	0	0	0	0	2	1	0	345	1	1.06	0.94;
	31	2	9.2	4.6	0	0	1	1	0	345	1	1.06	0.94;
	32	2	0	0	0	0	1	1	0	345	1	1.06	0.94;
	33	2	0	0	0	0	3	1	0	345	1	1.06	0.94;
	34	2	0	0	0	0	3	1	0	345	1	1.06	0.94;
	35	2	0	0	0	0	3	1	0	345	1	1.06	0.94;
	36	2	0	0	0	0	3	1	0	345	1	1.06	0.94;
	37	2	0	0	0	0	2	1	0	345	1	1.06	0.94;
	38	2	0	0	0	0	3	1	0	345	1	1.06	0.94;
	39	2	1104	250	0	0	1	1	0	345	1	1.06	0.94;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	30	520	270	400	140	1	100	1	1040	0	0	0	0	0	0	0	0	0	0	0	0;
	31	323	100	300	-100	1	100	1	646	0	0	0	0	0	0	0	0	0	0	0	0;
	32	362.5	225	300	150	1	100	1	725	0	0	0	0	0	0	0	0	0	0	0	0;
	33	326	125	250	0	1	100	1	652	0	0	0	0	0	0	0	0	0	0	0	0;
	34	254	83.5	167	0	1	100	1	508	0	0	0	0	0	0	0	0	0	0	0	0;
	35	343.5	100	300	-100	1	100	1	687	0	0	0	0	0	0	0	0	0	0	0	0;
	36	290	120	240	0	1	100	1	580	0	0	0	0	0	0	0	0	0	0	0	0;
	37	282	125	250	0	1	100	1	564	0	0	0	0	0	0	0	0	0	0	0	0;
	38	432.5	75	300	-150	1	100	1	865	0	0	0	0	0	0	0	0	0	0	0	0;
	39	550	100	300	-100	1	100	1	1100	0	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	2	0.0035	0.0411	0.6987	600	600	600	0	0	1	-360	360;
	1	39	0.001	0.025	0.75	1000	1000	1000	0	0	1	-360	360;
	2	3	0.0013	0.0151	0.2572	500	500	500	0	0	1	-360	360;
	2	25	0.007	0.0086	0.146	500	500	500	0	0	1	-360	360;
	2	30	0	0.0181	0	900	900	2500	1.025	0	1	-360	360;
	3	4	0.0013	0.0213	0.2214	500	500	500	0	0	1	-360	360;
	3	18	0.0011	0.0133	0.2138	500	500	500	0	0	1	-360	360;
	4	5	0.0008	0.0128	0.1342	600	600	600	0	0	1	-360	360;
	4	14	0.0008	0.0129	0.1382	500	500	500	0	0	1	-360	360;
	5	6	0.0002	0.0026	0.0434	1200	1200	1200	0	0	1	-360	360;
	5	8	0.0008	0.0112	0.1476	900	900	900	0	0	1	-360	360;
	6	7	0.0006	0.0092	0.113	900	900	900	0	0	1	-360	360;
	6	11	0.0007	0.0082	0.1389	480	480	480	0	0	1	-360	360;
	6	31	0	0.025	0	1800	1800	1800	1.07	0	1	-360	360;
	7	8	0.0004	0.0046	0.078	900	900	900	0	0	1	-360	360;
	8	9	0.0023	0.0363	0.3804	900	900	900	0	0	1	-360	360;
	9	39	0.001	0.025	1.2	900	900	900	0	0	1	-360	360;
	10	11	0.0004	0.0043	0.0729	600	600	600	0	0	1	-360	360;
	10	13	0.0004	0.0043	0.0729	600	600	600	0	0	1	-360	360;
	10	32	0	0.02	0	900	900	2500	1.07	0	1	-360	360;
	12	11	0.0016	0.0435	0	500	500	500	1.006	0	1	-360	360;
	12	13	0.0016	0.0435	0	500	500	500	1.006	0	1	-360	360;
	13	14	0.0009	0.0101	0.1723	600	600	600	0	0	1	-360	360;
	14	15	0.0018	0.0217	0.366	600	600	600	0	0	1	-360	360;
	15	16	0.0009	0.0094	0.171	600	600	600	0	0	1	-360	360;
	16	17	0.0007	0.0089	0.1342	600	600	600	0	0	1	-360	360;
	16	19	0.0016	0.0195	0.304	600	600	2500	0	0	1	-360	360;
	16	21	0.0008	0.0135	0.2548	600	600	600	0	0	1	-360	360;
	16	24	0.0003	0.0059	0.068	600	600	600	0	0	1	-360	360;
	17	18	0.0007	0.0082	0.1319	600	600	600	0	0	1	-360	360;
	17	27	0.0013	0.0173	0.3216	600	600	600	0	0	1	-360	360;
	19	20	0.0007	0.0138	0	900	900	2500	1.06	0	1	-360	360;
	19	33	0.0007	0.0142	0	900	900	2500	1.07	0	1	-360	360;
	20	34	0.0009	0.018	0	900	900	2500	1.009	0	1	-360	360;
	21	22	0.0008	0.014	0.2565	900	900	900	0	0	1	-360	360;
	22	23	0.0006	0.0096	0.1846	600	600	600	0	0	1	-360	360;
	22	35	0	0.0143	0	900	900	2500	1.025	0	1	-360	360;
	23	24	0.0022	0.035	0.361	600	600	600	0	0	1	-360	360;
	23	36	0.0005	0.0272	0	900	900	2500	0	0	1	-360	360;
	25	26	0.0032	0.0323	0.531	600	600	600	0	0	1	-360	360;
	25	37	0.0006	0.0232	0	900	900	2500	1.025	0	1	-360	360;
	26	27	0.0014	0.0147	0.2396	600	600	600	0	0	1	-360	360;
	26	28	0.0043	0.0474	0.7802	600	600	600	0	0	1	-360	360;
	26	29	0.0057	0.0625	1.029	600	600	600	0	0	1	-360	360;
	28	29	0.0014	0.0151	0.249	600	600	600	0	0	1	-360	360;
	29	38	0.0008	0.0156	0	1200	1200	2500	1.025	0	1	-360	360;
];

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	0	0	3	0	6.724778	0;
	2	0	0	3	0	14.707625	0;
	2	0	0	3	0	24.804734	0;
	2	0	0	3	0	34.844643	0;
	2	0	0	3	0	24.652994	0;
	2	0	0	3	0	32.306483	0;
	2	0	0	3	0	18.157477	0;
	2	0	0	3	0	31.550181	0;
	2	0	0	3	0	22.503168	0;
	2	0	0	3	0	27.434444	0;
];
