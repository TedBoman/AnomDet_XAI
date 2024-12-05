--
-- PostgreSQL database dump
--

-- Dumped from database version 15.8
-- Dumped by pg_dump version 15.8

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: my_data; Type: TABLE; Schema: public; Owner: Anomdet
--

CREATE TABLE public.my_data (
    "timestamp" bigint,
    load_1m numeric,
    load_5m numeric,
    load_15m numeric,
    sys_mem_swap_total bigint,
    sys_mem_swap_free bigint,
    sys_mem_free bigint,
    sys_mem_cache bigint,
    sys_mem_buffered bigint,
    sys_mem_available bigint,
    sys_mem_total bigint,
    sys_fork_rate numeric,
    sys_interrupt_rate numeric,
    sys_context_switch_rate numeric,
    sys_thermal numeric,
    disk_io_time numeric,
    disk_bytes_read numeric,
    disk_bytes_written numeric,
    disk_io_read numeric,
    disk_io_write numeric,
    cpu_iowait numeric,
    cpu_system numeric,
    cpu_user numeric,
    server_up bigint
);


ALTER TABLE public.my_data OWNER TO "Anomdet";

--
-- Data for Name: my_data; Type: TABLE DATA; Schema: public; Owner: Anomdet
--

COPY public.my_data ("timestamp", load_1m, load_5m, load_15m, sys_mem_swap_total, sys_mem_swap_free, sys_mem_free, sys_mem_cache, sys_mem_buffered, sys_mem_available, sys_mem_total, sys_fork_rate, sys_interrupt_rate, sys_context_switch_rate, sys_thermal, disk_io_time, disk_bytes_read, disk_bytes_written, disk_io_read, disk_io_write, cpu_iowait, cpu_system, cpu_user, server_up) FROM stdin;
0	0.02	0.05	0.06	34274799616	34274799616	23795159040	7048232960	1094365184	32358117376	33647591424	0.95	955.45	2403.75	0.2	0.0010000000000218	0.0	15769.6	0.0	1.85	0.0020000000000436	0.0270000000000436	0.0255000000001018	2
\.


--
-- PostgreSQL database dump complete
--

