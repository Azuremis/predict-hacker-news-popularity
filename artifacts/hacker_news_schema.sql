--
-- PostgreSQL database dump
--

-- Dumped from database version 17.0 (Debian 17.0-1.pgdg120+1)
-- Dumped by pg_dump version 17.4 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: hacker_news; Type: SCHEMA; Schema: -; Owner: zer4bab
--

CREATE SCHEMA hacker_news;


ALTER SCHEMA hacker_news OWNER TO zer4bab;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: items; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items OWNER TO zer4bab;

--
-- Name: items_by_month; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
)
PARTITION BY RANGE ("time");


ALTER TABLE hacker_news.items_by_month OWNER TO zer4bab;

--
-- Name: items_by_month_2006_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2006_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2006_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2006_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2006_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2006_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2007_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2007_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2007_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2008_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2008_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2008_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2009_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2009_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2009_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2010_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2010_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2010_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2011_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2011_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2011_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2012_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2012_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2012_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2013_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2013_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2013_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2014_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2014_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2014_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2015_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2015_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2015_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2016_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2016_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2016_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2017_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2017_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2017_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2018_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2018_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2018_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2019_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2019_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2019_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2020_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2020_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2020_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2021_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2021_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2021_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2022_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2022_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2022_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_10 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_11; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_11 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_11 OWNER TO zer4bab;

--
-- Name: items_by_month_2023_12; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2023_12 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2023_12 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_01; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_01 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_01 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_02; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_02 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_02 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_03; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_03 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_03 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_04; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_04 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_04 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_05; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_05 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_05 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_06; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_06 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_06 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_07; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_07 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_07 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_08; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_08 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_08 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_09; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_09 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_09 OWNER TO zer4bab;

--
-- Name: items_by_month_2024_10; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_month_2024_10 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_month_2024_10 OWNER TO zer4bab;

--
-- Name: items_by_month_YYYY_XX; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news."items_by_month_YYYY_XX" (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news."items_by_month_YYYY_XX" OWNER TO zer4bab;

--
-- Name: items_by_year; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
)
PARTITION BY RANGE ("time");


ALTER TABLE hacker_news.items_by_year OWNER TO zer4bab;

--
-- Name: items_by_year_2006; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2006 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2006 OWNER TO zer4bab;

--
-- Name: items_by_year_2007; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2007 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2007 OWNER TO zer4bab;

--
-- Name: items_by_year_2008; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2008 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2008 OWNER TO zer4bab;

--
-- Name: items_by_year_2009; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2009 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2009 OWNER TO zer4bab;

--
-- Name: items_by_year_2010; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2010 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2010 OWNER TO zer4bab;

--
-- Name: items_by_year_2011; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2011 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2011 OWNER TO zer4bab;

--
-- Name: items_by_year_2012; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2012 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2012 OWNER TO zer4bab;

--
-- Name: items_by_year_2013; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2013 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2013 OWNER TO zer4bab;

--
-- Name: items_by_year_2014; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2014 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2014 OWNER TO zer4bab;

--
-- Name: items_by_year_2015; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2015 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2015 OWNER TO zer4bab;

--
-- Name: items_by_year_2016; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2016 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2016 OWNER TO zer4bab;

--
-- Name: items_by_year_2017; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2017 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2017 OWNER TO zer4bab;

--
-- Name: items_by_year_2018; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2018 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2018 OWNER TO zer4bab;

--
-- Name: items_by_year_2019; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2019 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2019 OWNER TO zer4bab;

--
-- Name: items_by_year_2020; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2020 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2020 OWNER TO zer4bab;

--
-- Name: items_by_year_2021; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2021 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2021 OWNER TO zer4bab;

--
-- Name: items_by_year_2022; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2022 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2022 OWNER TO zer4bab;

--
-- Name: items_by_year_2023; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2023 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2023 OWNER TO zer4bab;

--
-- Name: items_by_year_2024; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.items_by_year_2024 (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news.items_by_year_2024 OWNER TO zer4bab;

--
-- Name: items_by_year_YYYY; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news."items_by_year_YYYY" (
    id integer NOT NULL,
    dead boolean,
    type character varying(20),
    by character varying(255),
    "time" timestamp without time zone,
    text text,
    parent integer,
    kids integer[],
    url character varying(255),
    score integer,
    title character varying(255),
    descendants integer
);


ALTER TABLE hacker_news."items_by_year_YYYY" OWNER TO zer4bab;

--
-- Name: users; Type: TABLE; Schema: hacker_news; Owner: zer4bab
--

CREATE TABLE hacker_news.users (
    id character varying(255) NOT NULL,
    created timestamp without time zone,
    karma integer,
    about text,
    submitted integer[]
);


ALTER TABLE hacker_news.users OWNER TO zer4bab;

--
-- Name: items_by_month_2006_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2006_10 FOR VALUES FROM ('2006-10-01 00:00:00') TO ('2006-11-01 00:00:00');


--
-- Name: items_by_month_2006_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2006_12 FOR VALUES FROM ('2006-12-01 00:00:00') TO ('2007-01-01 00:00:00');


--
-- Name: items_by_month_2007_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_02 FOR VALUES FROM ('2007-02-01 00:00:00') TO ('2007-03-01 00:00:00');


--
-- Name: items_by_month_2007_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_03 FOR VALUES FROM ('2007-03-01 00:00:00') TO ('2007-04-01 00:00:00');


--
-- Name: items_by_month_2007_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_04 FOR VALUES FROM ('2007-04-01 00:00:00') TO ('2007-05-01 00:00:00');


--
-- Name: items_by_month_2007_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_05 FOR VALUES FROM ('2007-05-01 00:00:00') TO ('2007-06-01 00:00:00');


--
-- Name: items_by_month_2007_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_06 FOR VALUES FROM ('2007-06-01 00:00:00') TO ('2007-07-01 00:00:00');


--
-- Name: items_by_month_2007_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_07 FOR VALUES FROM ('2007-07-01 00:00:00') TO ('2007-08-01 00:00:00');


--
-- Name: items_by_month_2007_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_08 FOR VALUES FROM ('2007-08-01 00:00:00') TO ('2007-09-01 00:00:00');


--
-- Name: items_by_month_2007_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_09 FOR VALUES FROM ('2007-09-01 00:00:00') TO ('2007-10-01 00:00:00');


--
-- Name: items_by_month_2007_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_10 FOR VALUES FROM ('2007-10-01 00:00:00') TO ('2007-11-01 00:00:00');


--
-- Name: items_by_month_2007_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_11 FOR VALUES FROM ('2007-11-01 00:00:00') TO ('2007-12-01 00:00:00');


--
-- Name: items_by_month_2007_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2007_12 FOR VALUES FROM ('2007-12-01 00:00:00') TO ('2008-01-01 00:00:00');


--
-- Name: items_by_month_2008_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_01 FOR VALUES FROM ('2008-01-01 00:00:00') TO ('2008-02-01 00:00:00');


--
-- Name: items_by_month_2008_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_02 FOR VALUES FROM ('2008-02-01 00:00:00') TO ('2008-03-01 00:00:00');


--
-- Name: items_by_month_2008_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_03 FOR VALUES FROM ('2008-03-01 00:00:00') TO ('2008-04-01 00:00:00');


--
-- Name: items_by_month_2008_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_04 FOR VALUES FROM ('2008-04-01 00:00:00') TO ('2008-05-01 00:00:00');


--
-- Name: items_by_month_2008_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_05 FOR VALUES FROM ('2008-05-01 00:00:00') TO ('2008-06-01 00:00:00');


--
-- Name: items_by_month_2008_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_06 FOR VALUES FROM ('2008-06-01 00:00:00') TO ('2008-07-01 00:00:00');


--
-- Name: items_by_month_2008_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_07 FOR VALUES FROM ('2008-07-01 00:00:00') TO ('2008-08-01 00:00:00');


--
-- Name: items_by_month_2008_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_08 FOR VALUES FROM ('2008-08-01 00:00:00') TO ('2008-09-01 00:00:00');


--
-- Name: items_by_month_2008_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_09 FOR VALUES FROM ('2008-09-01 00:00:00') TO ('2008-10-01 00:00:00');


--
-- Name: items_by_month_2008_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_10 FOR VALUES FROM ('2008-10-01 00:00:00') TO ('2008-11-01 00:00:00');


--
-- Name: items_by_month_2008_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_11 FOR VALUES FROM ('2008-11-01 00:00:00') TO ('2008-12-01 00:00:00');


--
-- Name: items_by_month_2008_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2008_12 FOR VALUES FROM ('2008-12-01 00:00:00') TO ('2009-01-01 00:00:00');


--
-- Name: items_by_month_2009_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_01 FOR VALUES FROM ('2009-01-01 00:00:00') TO ('2009-02-01 00:00:00');


--
-- Name: items_by_month_2009_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_02 FOR VALUES FROM ('2009-02-01 00:00:00') TO ('2009-03-01 00:00:00');


--
-- Name: items_by_month_2009_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_03 FOR VALUES FROM ('2009-03-01 00:00:00') TO ('2009-04-01 00:00:00');


--
-- Name: items_by_month_2009_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_04 FOR VALUES FROM ('2009-04-01 00:00:00') TO ('2009-05-01 00:00:00');


--
-- Name: items_by_month_2009_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_05 FOR VALUES FROM ('2009-05-01 00:00:00') TO ('2009-06-01 00:00:00');


--
-- Name: items_by_month_2009_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_06 FOR VALUES FROM ('2009-06-01 00:00:00') TO ('2009-07-01 00:00:00');


--
-- Name: items_by_month_2009_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_07 FOR VALUES FROM ('2009-07-01 00:00:00') TO ('2009-08-01 00:00:00');


--
-- Name: items_by_month_2009_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_08 FOR VALUES FROM ('2009-08-01 00:00:00') TO ('2009-09-01 00:00:00');


--
-- Name: items_by_month_2009_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_09 FOR VALUES FROM ('2009-09-01 00:00:00') TO ('2009-10-01 00:00:00');


--
-- Name: items_by_month_2009_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_10 FOR VALUES FROM ('2009-10-01 00:00:00') TO ('2009-11-01 00:00:00');


--
-- Name: items_by_month_2009_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_11 FOR VALUES FROM ('2009-11-01 00:00:00') TO ('2009-12-01 00:00:00');


--
-- Name: items_by_month_2009_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2009_12 FOR VALUES FROM ('2009-12-01 00:00:00') TO ('2010-01-01 00:00:00');


--
-- Name: items_by_month_2010_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_01 FOR VALUES FROM ('2010-01-01 00:00:00') TO ('2010-02-01 00:00:00');


--
-- Name: items_by_month_2010_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_02 FOR VALUES FROM ('2010-02-01 00:00:00') TO ('2010-03-01 00:00:00');


--
-- Name: items_by_month_2010_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_03 FOR VALUES FROM ('2010-03-01 00:00:00') TO ('2010-04-01 00:00:00');


--
-- Name: items_by_month_2010_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_04 FOR VALUES FROM ('2010-04-01 00:00:00') TO ('2010-05-01 00:00:00');


--
-- Name: items_by_month_2010_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_05 FOR VALUES FROM ('2010-05-01 00:00:00') TO ('2010-06-01 00:00:00');


--
-- Name: items_by_month_2010_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_06 FOR VALUES FROM ('2010-06-01 00:00:00') TO ('2010-07-01 00:00:00');


--
-- Name: items_by_month_2010_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_07 FOR VALUES FROM ('2010-07-01 00:00:00') TO ('2010-08-01 00:00:00');


--
-- Name: items_by_month_2010_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_08 FOR VALUES FROM ('2010-08-01 00:00:00') TO ('2010-09-01 00:00:00');


--
-- Name: items_by_month_2010_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_09 FOR VALUES FROM ('2010-09-01 00:00:00') TO ('2010-10-01 00:00:00');


--
-- Name: items_by_month_2010_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_10 FOR VALUES FROM ('2010-10-01 00:00:00') TO ('2010-11-01 00:00:00');


--
-- Name: items_by_month_2010_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_11 FOR VALUES FROM ('2010-11-01 00:00:00') TO ('2010-12-01 00:00:00');


--
-- Name: items_by_month_2010_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2010_12 FOR VALUES FROM ('2010-12-01 00:00:00') TO ('2011-01-01 00:00:00');


--
-- Name: items_by_month_2011_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_01 FOR VALUES FROM ('2011-01-01 00:00:00') TO ('2011-02-01 00:00:00');


--
-- Name: items_by_month_2011_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_02 FOR VALUES FROM ('2011-02-01 00:00:00') TO ('2011-03-01 00:00:00');


--
-- Name: items_by_month_2011_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_03 FOR VALUES FROM ('2011-03-01 00:00:00') TO ('2011-04-01 00:00:00');


--
-- Name: items_by_month_2011_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_04 FOR VALUES FROM ('2011-04-01 00:00:00') TO ('2011-05-01 00:00:00');


--
-- Name: items_by_month_2011_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_05 FOR VALUES FROM ('2011-05-01 00:00:00') TO ('2011-06-01 00:00:00');


--
-- Name: items_by_month_2011_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_06 FOR VALUES FROM ('2011-06-01 00:00:00') TO ('2011-07-01 00:00:00');


--
-- Name: items_by_month_2011_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_07 FOR VALUES FROM ('2011-07-01 00:00:00') TO ('2011-08-01 00:00:00');


--
-- Name: items_by_month_2011_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_08 FOR VALUES FROM ('2011-08-01 00:00:00') TO ('2011-09-01 00:00:00');


--
-- Name: items_by_month_2011_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_09 FOR VALUES FROM ('2011-09-01 00:00:00') TO ('2011-10-01 00:00:00');


--
-- Name: items_by_month_2011_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_10 FOR VALUES FROM ('2011-10-01 00:00:00') TO ('2011-11-01 00:00:00');


--
-- Name: items_by_month_2011_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_11 FOR VALUES FROM ('2011-11-01 00:00:00') TO ('2011-12-01 00:00:00');


--
-- Name: items_by_month_2011_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2011_12 FOR VALUES FROM ('2011-12-01 00:00:00') TO ('2012-01-01 00:00:00');


--
-- Name: items_by_month_2012_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_01 FOR VALUES FROM ('2012-01-01 00:00:00') TO ('2012-02-01 00:00:00');


--
-- Name: items_by_month_2012_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_02 FOR VALUES FROM ('2012-02-01 00:00:00') TO ('2012-03-01 00:00:00');


--
-- Name: items_by_month_2012_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_03 FOR VALUES FROM ('2012-03-01 00:00:00') TO ('2012-04-01 00:00:00');


--
-- Name: items_by_month_2012_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_04 FOR VALUES FROM ('2012-04-01 00:00:00') TO ('2012-05-01 00:00:00');


--
-- Name: items_by_month_2012_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_05 FOR VALUES FROM ('2012-05-01 00:00:00') TO ('2012-06-01 00:00:00');


--
-- Name: items_by_month_2012_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_06 FOR VALUES FROM ('2012-06-01 00:00:00') TO ('2012-07-01 00:00:00');


--
-- Name: items_by_month_2012_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_07 FOR VALUES FROM ('2012-07-01 00:00:00') TO ('2012-08-01 00:00:00');


--
-- Name: items_by_month_2012_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_08 FOR VALUES FROM ('2012-08-01 00:00:00') TO ('2012-09-01 00:00:00');


--
-- Name: items_by_month_2012_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_09 FOR VALUES FROM ('2012-09-01 00:00:00') TO ('2012-10-01 00:00:00');


--
-- Name: items_by_month_2012_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_10 FOR VALUES FROM ('2012-10-01 00:00:00') TO ('2012-11-01 00:00:00');


--
-- Name: items_by_month_2012_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_11 FOR VALUES FROM ('2012-11-01 00:00:00') TO ('2012-12-01 00:00:00');


--
-- Name: items_by_month_2012_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2012_12 FOR VALUES FROM ('2012-12-01 00:00:00') TO ('2013-01-01 00:00:00');


--
-- Name: items_by_month_2013_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_01 FOR VALUES FROM ('2013-01-01 00:00:00') TO ('2013-02-01 00:00:00');


--
-- Name: items_by_month_2013_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_02 FOR VALUES FROM ('2013-02-01 00:00:00') TO ('2013-03-01 00:00:00');


--
-- Name: items_by_month_2013_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_03 FOR VALUES FROM ('2013-03-01 00:00:00') TO ('2013-04-01 00:00:00');


--
-- Name: items_by_month_2013_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_04 FOR VALUES FROM ('2013-04-01 00:00:00') TO ('2013-05-01 00:00:00');


--
-- Name: items_by_month_2013_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_05 FOR VALUES FROM ('2013-05-01 00:00:00') TO ('2013-06-01 00:00:00');


--
-- Name: items_by_month_2013_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_06 FOR VALUES FROM ('2013-06-01 00:00:00') TO ('2013-07-01 00:00:00');


--
-- Name: items_by_month_2013_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_07 FOR VALUES FROM ('2013-07-01 00:00:00') TO ('2013-08-01 00:00:00');


--
-- Name: items_by_month_2013_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_08 FOR VALUES FROM ('2013-08-01 00:00:00') TO ('2013-09-01 00:00:00');


--
-- Name: items_by_month_2013_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_09 FOR VALUES FROM ('2013-09-01 00:00:00') TO ('2013-10-01 00:00:00');


--
-- Name: items_by_month_2013_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_10 FOR VALUES FROM ('2013-10-01 00:00:00') TO ('2013-11-01 00:00:00');


--
-- Name: items_by_month_2013_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_11 FOR VALUES FROM ('2013-11-01 00:00:00') TO ('2013-12-01 00:00:00');


--
-- Name: items_by_month_2013_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2013_12 FOR VALUES FROM ('2013-12-01 00:00:00') TO ('2014-01-01 00:00:00');


--
-- Name: items_by_month_2014_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_01 FOR VALUES FROM ('2014-01-01 00:00:00') TO ('2014-02-01 00:00:00');


--
-- Name: items_by_month_2014_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_02 FOR VALUES FROM ('2014-02-01 00:00:00') TO ('2014-03-01 00:00:00');


--
-- Name: items_by_month_2014_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_03 FOR VALUES FROM ('2014-03-01 00:00:00') TO ('2014-04-01 00:00:00');


--
-- Name: items_by_month_2014_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_04 FOR VALUES FROM ('2014-04-01 00:00:00') TO ('2014-05-01 00:00:00');


--
-- Name: items_by_month_2014_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_05 FOR VALUES FROM ('2014-05-01 00:00:00') TO ('2014-06-01 00:00:00');


--
-- Name: items_by_month_2014_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_06 FOR VALUES FROM ('2014-06-01 00:00:00') TO ('2014-07-01 00:00:00');


--
-- Name: items_by_month_2014_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_07 FOR VALUES FROM ('2014-07-01 00:00:00') TO ('2014-08-01 00:00:00');


--
-- Name: items_by_month_2014_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_08 FOR VALUES FROM ('2014-08-01 00:00:00') TO ('2014-09-01 00:00:00');


--
-- Name: items_by_month_2014_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_09 FOR VALUES FROM ('2014-09-01 00:00:00') TO ('2014-10-01 00:00:00');


--
-- Name: items_by_month_2014_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_10 FOR VALUES FROM ('2014-10-01 00:00:00') TO ('2014-11-01 00:00:00');


--
-- Name: items_by_month_2014_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_11 FOR VALUES FROM ('2014-11-01 00:00:00') TO ('2014-12-01 00:00:00');


--
-- Name: items_by_month_2014_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2014_12 FOR VALUES FROM ('2014-12-01 00:00:00') TO ('2015-01-01 00:00:00');


--
-- Name: items_by_month_2015_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_01 FOR VALUES FROM ('2015-01-01 00:00:00') TO ('2015-02-01 00:00:00');


--
-- Name: items_by_month_2015_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_02 FOR VALUES FROM ('2015-02-01 00:00:00') TO ('2015-03-01 00:00:00');


--
-- Name: items_by_month_2015_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_03 FOR VALUES FROM ('2015-03-01 00:00:00') TO ('2015-04-01 00:00:00');


--
-- Name: items_by_month_2015_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_04 FOR VALUES FROM ('2015-04-01 00:00:00') TO ('2015-05-01 00:00:00');


--
-- Name: items_by_month_2015_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_05 FOR VALUES FROM ('2015-05-01 00:00:00') TO ('2015-06-01 00:00:00');


--
-- Name: items_by_month_2015_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_06 FOR VALUES FROM ('2015-06-01 00:00:00') TO ('2015-07-01 00:00:00');


--
-- Name: items_by_month_2015_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_07 FOR VALUES FROM ('2015-07-01 00:00:00') TO ('2015-08-01 00:00:00');


--
-- Name: items_by_month_2015_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_08 FOR VALUES FROM ('2015-08-01 00:00:00') TO ('2015-09-01 00:00:00');


--
-- Name: items_by_month_2015_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_09 FOR VALUES FROM ('2015-09-01 00:00:00') TO ('2015-10-01 00:00:00');


--
-- Name: items_by_month_2015_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_10 FOR VALUES FROM ('2015-10-01 00:00:00') TO ('2015-11-01 00:00:00');


--
-- Name: items_by_month_2015_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_11 FOR VALUES FROM ('2015-11-01 00:00:00') TO ('2015-12-01 00:00:00');


--
-- Name: items_by_month_2015_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2015_12 FOR VALUES FROM ('2015-12-01 00:00:00') TO ('2016-01-01 00:00:00');


--
-- Name: items_by_month_2016_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_01 FOR VALUES FROM ('2016-01-01 00:00:00') TO ('2016-02-01 00:00:00');


--
-- Name: items_by_month_2016_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_02 FOR VALUES FROM ('2016-02-01 00:00:00') TO ('2016-03-01 00:00:00');


--
-- Name: items_by_month_2016_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_03 FOR VALUES FROM ('2016-03-01 00:00:00') TO ('2016-04-01 00:00:00');


--
-- Name: items_by_month_2016_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_04 FOR VALUES FROM ('2016-04-01 00:00:00') TO ('2016-05-01 00:00:00');


--
-- Name: items_by_month_2016_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_05 FOR VALUES FROM ('2016-05-01 00:00:00') TO ('2016-06-01 00:00:00');


--
-- Name: items_by_month_2016_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_06 FOR VALUES FROM ('2016-06-01 00:00:00') TO ('2016-07-01 00:00:00');


--
-- Name: items_by_month_2016_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_07 FOR VALUES FROM ('2016-07-01 00:00:00') TO ('2016-08-01 00:00:00');


--
-- Name: items_by_month_2016_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_08 FOR VALUES FROM ('2016-08-01 00:00:00') TO ('2016-09-01 00:00:00');


--
-- Name: items_by_month_2016_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_09 FOR VALUES FROM ('2016-09-01 00:00:00') TO ('2016-10-01 00:00:00');


--
-- Name: items_by_month_2016_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_10 FOR VALUES FROM ('2016-10-01 00:00:00') TO ('2016-11-01 00:00:00');


--
-- Name: items_by_month_2016_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_11 FOR VALUES FROM ('2016-11-01 00:00:00') TO ('2016-12-01 00:00:00');


--
-- Name: items_by_month_2016_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2016_12 FOR VALUES FROM ('2016-12-01 00:00:00') TO ('2017-01-01 00:00:00');


--
-- Name: items_by_month_2017_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_01 FOR VALUES FROM ('2017-01-01 00:00:00') TO ('2017-02-01 00:00:00');


--
-- Name: items_by_month_2017_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_02 FOR VALUES FROM ('2017-02-01 00:00:00') TO ('2017-03-01 00:00:00');


--
-- Name: items_by_month_2017_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_03 FOR VALUES FROM ('2017-03-01 00:00:00') TO ('2017-04-01 00:00:00');


--
-- Name: items_by_month_2017_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_04 FOR VALUES FROM ('2017-04-01 00:00:00') TO ('2017-05-01 00:00:00');


--
-- Name: items_by_month_2017_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_05 FOR VALUES FROM ('2017-05-01 00:00:00') TO ('2017-06-01 00:00:00');


--
-- Name: items_by_month_2017_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_06 FOR VALUES FROM ('2017-06-01 00:00:00') TO ('2017-07-01 00:00:00');


--
-- Name: items_by_month_2017_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_07 FOR VALUES FROM ('2017-07-01 00:00:00') TO ('2017-08-01 00:00:00');


--
-- Name: items_by_month_2017_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_08 FOR VALUES FROM ('2017-08-01 00:00:00') TO ('2017-09-01 00:00:00');


--
-- Name: items_by_month_2017_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_09 FOR VALUES FROM ('2017-09-01 00:00:00') TO ('2017-10-01 00:00:00');


--
-- Name: items_by_month_2017_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_10 FOR VALUES FROM ('2017-10-01 00:00:00') TO ('2017-11-01 00:00:00');


--
-- Name: items_by_month_2017_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_11 FOR VALUES FROM ('2017-11-01 00:00:00') TO ('2017-12-01 00:00:00');


--
-- Name: items_by_month_2017_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2017_12 FOR VALUES FROM ('2017-12-01 00:00:00') TO ('2018-01-01 00:00:00');


--
-- Name: items_by_month_2018_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_01 FOR VALUES FROM ('2018-01-01 00:00:00') TO ('2018-02-01 00:00:00');


--
-- Name: items_by_month_2018_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_02 FOR VALUES FROM ('2018-02-01 00:00:00') TO ('2018-03-01 00:00:00');


--
-- Name: items_by_month_2018_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_03 FOR VALUES FROM ('2018-03-01 00:00:00') TO ('2018-04-01 00:00:00');


--
-- Name: items_by_month_2018_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_04 FOR VALUES FROM ('2018-04-01 00:00:00') TO ('2018-05-01 00:00:00');


--
-- Name: items_by_month_2018_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_05 FOR VALUES FROM ('2018-05-01 00:00:00') TO ('2018-06-01 00:00:00');


--
-- Name: items_by_month_2018_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_06 FOR VALUES FROM ('2018-06-01 00:00:00') TO ('2018-07-01 00:00:00');


--
-- Name: items_by_month_2018_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_07 FOR VALUES FROM ('2018-07-01 00:00:00') TO ('2018-08-01 00:00:00');


--
-- Name: items_by_month_2018_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_08 FOR VALUES FROM ('2018-08-01 00:00:00') TO ('2018-09-01 00:00:00');


--
-- Name: items_by_month_2018_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_09 FOR VALUES FROM ('2018-09-01 00:00:00') TO ('2018-10-01 00:00:00');


--
-- Name: items_by_month_2018_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_10 FOR VALUES FROM ('2018-10-01 00:00:00') TO ('2018-11-01 00:00:00');


--
-- Name: items_by_month_2018_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_11 FOR VALUES FROM ('2018-11-01 00:00:00') TO ('2018-12-01 00:00:00');


--
-- Name: items_by_month_2018_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2018_12 FOR VALUES FROM ('2018-12-01 00:00:00') TO ('2019-01-01 00:00:00');


--
-- Name: items_by_month_2019_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_01 FOR VALUES FROM ('2019-01-01 00:00:00') TO ('2019-02-01 00:00:00');


--
-- Name: items_by_month_2019_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_02 FOR VALUES FROM ('2019-02-01 00:00:00') TO ('2019-03-01 00:00:00');


--
-- Name: items_by_month_2019_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_03 FOR VALUES FROM ('2019-03-01 00:00:00') TO ('2019-04-01 00:00:00');


--
-- Name: items_by_month_2019_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_04 FOR VALUES FROM ('2019-04-01 00:00:00') TO ('2019-05-01 00:00:00');


--
-- Name: items_by_month_2019_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_05 FOR VALUES FROM ('2019-05-01 00:00:00') TO ('2019-06-01 00:00:00');


--
-- Name: items_by_month_2019_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_06 FOR VALUES FROM ('2019-06-01 00:00:00') TO ('2019-07-01 00:00:00');


--
-- Name: items_by_month_2019_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_07 FOR VALUES FROM ('2019-07-01 00:00:00') TO ('2019-08-01 00:00:00');


--
-- Name: items_by_month_2019_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_08 FOR VALUES FROM ('2019-08-01 00:00:00') TO ('2019-09-01 00:00:00');


--
-- Name: items_by_month_2019_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_09 FOR VALUES FROM ('2019-09-01 00:00:00') TO ('2019-10-01 00:00:00');


--
-- Name: items_by_month_2019_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_10 FOR VALUES FROM ('2019-10-01 00:00:00') TO ('2019-11-01 00:00:00');


--
-- Name: items_by_month_2019_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_11 FOR VALUES FROM ('2019-11-01 00:00:00') TO ('2019-12-01 00:00:00');


--
-- Name: items_by_month_2019_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2019_12 FOR VALUES FROM ('2019-12-01 00:00:00') TO ('2020-01-01 00:00:00');


--
-- Name: items_by_month_2020_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_01 FOR VALUES FROM ('2020-01-01 00:00:00') TO ('2020-02-01 00:00:00');


--
-- Name: items_by_month_2020_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_02 FOR VALUES FROM ('2020-02-01 00:00:00') TO ('2020-03-01 00:00:00');


--
-- Name: items_by_month_2020_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_03 FOR VALUES FROM ('2020-03-01 00:00:00') TO ('2020-04-01 00:00:00');


--
-- Name: items_by_month_2020_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_04 FOR VALUES FROM ('2020-04-01 00:00:00') TO ('2020-05-01 00:00:00');


--
-- Name: items_by_month_2020_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_05 FOR VALUES FROM ('2020-05-01 00:00:00') TO ('2020-06-01 00:00:00');


--
-- Name: items_by_month_2020_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_06 FOR VALUES FROM ('2020-06-01 00:00:00') TO ('2020-07-01 00:00:00');


--
-- Name: items_by_month_2020_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_07 FOR VALUES FROM ('2020-07-01 00:00:00') TO ('2020-08-01 00:00:00');


--
-- Name: items_by_month_2020_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_08 FOR VALUES FROM ('2020-08-01 00:00:00') TO ('2020-09-01 00:00:00');


--
-- Name: items_by_month_2020_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_09 FOR VALUES FROM ('2020-09-01 00:00:00') TO ('2020-10-01 00:00:00');


--
-- Name: items_by_month_2020_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_10 FOR VALUES FROM ('2020-10-01 00:00:00') TO ('2020-11-01 00:00:00');


--
-- Name: items_by_month_2020_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_11 FOR VALUES FROM ('2020-11-01 00:00:00') TO ('2020-12-01 00:00:00');


--
-- Name: items_by_month_2020_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2020_12 FOR VALUES FROM ('2020-12-01 00:00:00') TO ('2021-01-01 00:00:00');


--
-- Name: items_by_month_2021_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_01 FOR VALUES FROM ('2021-01-01 00:00:00') TO ('2021-02-01 00:00:00');


--
-- Name: items_by_month_2021_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_02 FOR VALUES FROM ('2021-02-01 00:00:00') TO ('2021-03-01 00:00:00');


--
-- Name: items_by_month_2021_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_03 FOR VALUES FROM ('2021-03-01 00:00:00') TO ('2021-04-01 00:00:00');


--
-- Name: items_by_month_2021_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_04 FOR VALUES FROM ('2021-04-01 00:00:00') TO ('2021-05-01 00:00:00');


--
-- Name: items_by_month_2021_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_05 FOR VALUES FROM ('2021-05-01 00:00:00') TO ('2021-06-01 00:00:00');


--
-- Name: items_by_month_2021_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_06 FOR VALUES FROM ('2021-06-01 00:00:00') TO ('2021-07-01 00:00:00');


--
-- Name: items_by_month_2021_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_07 FOR VALUES FROM ('2021-07-01 00:00:00') TO ('2021-08-01 00:00:00');


--
-- Name: items_by_month_2021_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_08 FOR VALUES FROM ('2021-08-01 00:00:00') TO ('2021-09-01 00:00:00');


--
-- Name: items_by_month_2021_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_09 FOR VALUES FROM ('2021-09-01 00:00:00') TO ('2021-10-01 00:00:00');


--
-- Name: items_by_month_2021_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_10 FOR VALUES FROM ('2021-10-01 00:00:00') TO ('2021-11-01 00:00:00');


--
-- Name: items_by_month_2021_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_11 FOR VALUES FROM ('2021-11-01 00:00:00') TO ('2021-12-01 00:00:00');


--
-- Name: items_by_month_2021_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2021_12 FOR VALUES FROM ('2021-12-01 00:00:00') TO ('2022-01-01 00:00:00');


--
-- Name: items_by_month_2022_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_01 FOR VALUES FROM ('2022-01-01 00:00:00') TO ('2022-02-01 00:00:00');


--
-- Name: items_by_month_2022_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_02 FOR VALUES FROM ('2022-02-01 00:00:00') TO ('2022-03-01 00:00:00');


--
-- Name: items_by_month_2022_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_03 FOR VALUES FROM ('2022-03-01 00:00:00') TO ('2022-04-01 00:00:00');


--
-- Name: items_by_month_2022_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_04 FOR VALUES FROM ('2022-04-01 00:00:00') TO ('2022-05-01 00:00:00');


--
-- Name: items_by_month_2022_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_05 FOR VALUES FROM ('2022-05-01 00:00:00') TO ('2022-06-01 00:00:00');


--
-- Name: items_by_month_2022_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_06 FOR VALUES FROM ('2022-06-01 00:00:00') TO ('2022-07-01 00:00:00');


--
-- Name: items_by_month_2022_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_07 FOR VALUES FROM ('2022-07-01 00:00:00') TO ('2022-08-01 00:00:00');


--
-- Name: items_by_month_2022_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_08 FOR VALUES FROM ('2022-08-01 00:00:00') TO ('2022-09-01 00:00:00');


--
-- Name: items_by_month_2022_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_09 FOR VALUES FROM ('2022-09-01 00:00:00') TO ('2022-10-01 00:00:00');


--
-- Name: items_by_month_2022_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_10 FOR VALUES FROM ('2022-10-01 00:00:00') TO ('2022-11-01 00:00:00');


--
-- Name: items_by_month_2022_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_11 FOR VALUES FROM ('2022-11-01 00:00:00') TO ('2022-12-01 00:00:00');


--
-- Name: items_by_month_2022_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2022_12 FOR VALUES FROM ('2022-12-01 00:00:00') TO ('2023-01-01 00:00:00');


--
-- Name: items_by_month_2023_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_01 FOR VALUES FROM ('2023-01-01 00:00:00') TO ('2023-02-01 00:00:00');


--
-- Name: items_by_month_2023_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_02 FOR VALUES FROM ('2023-02-01 00:00:00') TO ('2023-03-01 00:00:00');


--
-- Name: items_by_month_2023_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_03 FOR VALUES FROM ('2023-03-01 00:00:00') TO ('2023-04-01 00:00:00');


--
-- Name: items_by_month_2023_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_04 FOR VALUES FROM ('2023-04-01 00:00:00') TO ('2023-05-01 00:00:00');


--
-- Name: items_by_month_2023_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_05 FOR VALUES FROM ('2023-05-01 00:00:00') TO ('2023-06-01 00:00:00');


--
-- Name: items_by_month_2023_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_06 FOR VALUES FROM ('2023-06-01 00:00:00') TO ('2023-07-01 00:00:00');


--
-- Name: items_by_month_2023_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_07 FOR VALUES FROM ('2023-07-01 00:00:00') TO ('2023-08-01 00:00:00');


--
-- Name: items_by_month_2023_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_08 FOR VALUES FROM ('2023-08-01 00:00:00') TO ('2023-09-01 00:00:00');


--
-- Name: items_by_month_2023_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_09 FOR VALUES FROM ('2023-09-01 00:00:00') TO ('2023-10-01 00:00:00');


--
-- Name: items_by_month_2023_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_10 FOR VALUES FROM ('2023-10-01 00:00:00') TO ('2023-11-01 00:00:00');


--
-- Name: items_by_month_2023_11; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_11 FOR VALUES FROM ('2023-11-01 00:00:00') TO ('2023-12-01 00:00:00');


--
-- Name: items_by_month_2023_12; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2023_12 FOR VALUES FROM ('2023-12-01 00:00:00') TO ('2024-01-01 00:00:00');


--
-- Name: items_by_month_2024_01; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_01 FOR VALUES FROM ('2024-01-01 00:00:00') TO ('2024-02-01 00:00:00');


--
-- Name: items_by_month_2024_02; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_02 FOR VALUES FROM ('2024-02-01 00:00:00') TO ('2024-03-01 00:00:00');


--
-- Name: items_by_month_2024_03; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_03 FOR VALUES FROM ('2024-03-01 00:00:00') TO ('2024-04-01 00:00:00');


--
-- Name: items_by_month_2024_04; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_04 FOR VALUES FROM ('2024-04-01 00:00:00') TO ('2024-05-01 00:00:00');


--
-- Name: items_by_month_2024_05; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_05 FOR VALUES FROM ('2024-05-01 00:00:00') TO ('2024-06-01 00:00:00');


--
-- Name: items_by_month_2024_06; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_06 FOR VALUES FROM ('2024-06-01 00:00:00') TO ('2024-07-01 00:00:00');


--
-- Name: items_by_month_2024_07; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_07 FOR VALUES FROM ('2024-07-01 00:00:00') TO ('2024-08-01 00:00:00');


--
-- Name: items_by_month_2024_08; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_08 FOR VALUES FROM ('2024-08-01 00:00:00') TO ('2024-09-01 00:00:00');


--
-- Name: items_by_month_2024_09; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_09 FOR VALUES FROM ('2024-09-01 00:00:00') TO ('2024-10-01 00:00:00');


--
-- Name: items_by_month_2024_10; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news.items_by_month_2024_10 FOR VALUES FROM ('2024-10-01 00:00:00') TO ('2024-11-01 00:00:00');


--
-- Name: items_by_month_YYYY_XX; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_month ATTACH PARTITION hacker_news."items_by_month_YYYY_XX" DEFAULT;


--
-- Name: items_by_year_2006; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2006 FOR VALUES FROM ('2006-01-01 00:00:00') TO ('2007-01-01 00:00:00');


--
-- Name: items_by_year_2007; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2007 FOR VALUES FROM ('2007-01-01 00:00:00') TO ('2008-01-01 00:00:00');


--
-- Name: items_by_year_2008; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2008 FOR VALUES FROM ('2008-01-01 00:00:00') TO ('2009-01-01 00:00:00');


--
-- Name: items_by_year_2009; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2009 FOR VALUES FROM ('2009-01-01 00:00:00') TO ('2010-01-01 00:00:00');


--
-- Name: items_by_year_2010; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2010 FOR VALUES FROM ('2010-01-01 00:00:00') TO ('2011-01-01 00:00:00');


--
-- Name: items_by_year_2011; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2011 FOR VALUES FROM ('2011-01-01 00:00:00') TO ('2012-01-01 00:00:00');


--
-- Name: items_by_year_2012; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2012 FOR VALUES FROM ('2012-01-01 00:00:00') TO ('2013-01-01 00:00:00');


--
-- Name: items_by_year_2013; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2013 FOR VALUES FROM ('2013-01-01 00:00:00') TO ('2014-01-01 00:00:00');


--
-- Name: items_by_year_2014; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2014 FOR VALUES FROM ('2014-01-01 00:00:00') TO ('2015-01-01 00:00:00');


--
-- Name: items_by_year_2015; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2015 FOR VALUES FROM ('2015-01-01 00:00:00') TO ('2016-01-01 00:00:00');


--
-- Name: items_by_year_2016; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2016 FOR VALUES FROM ('2016-01-01 00:00:00') TO ('2017-01-01 00:00:00');


--
-- Name: items_by_year_2017; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2017 FOR VALUES FROM ('2017-01-01 00:00:00') TO ('2018-01-01 00:00:00');


--
-- Name: items_by_year_2018; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2018 FOR VALUES FROM ('2018-01-01 00:00:00') TO ('2019-01-01 00:00:00');


--
-- Name: items_by_year_2019; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2019 FOR VALUES FROM ('2019-01-01 00:00:00') TO ('2020-01-01 00:00:00');


--
-- Name: items_by_year_2020; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2020 FOR VALUES FROM ('2020-01-01 00:00:00') TO ('2021-01-01 00:00:00');


--
-- Name: items_by_year_2021; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2021 FOR VALUES FROM ('2021-01-01 00:00:00') TO ('2022-01-01 00:00:00');


--
-- Name: items_by_year_2022; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2022 FOR VALUES FROM ('2022-01-01 00:00:00') TO ('2023-01-01 00:00:00');


--
-- Name: items_by_year_2023; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2023 FOR VALUES FROM ('2023-01-01 00:00:00') TO ('2024-01-01 00:00:00');


--
-- Name: items_by_year_2024; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news.items_by_year_2024 FOR VALUES FROM ('2024-01-01 00:00:00') TO ('2025-01-01 00:00:00');


--
-- Name: items_by_year_YYYY; Type: TABLE ATTACH; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items_by_year ATTACH PARTITION hacker_news."items_by_year_YYYY" DEFAULT;


--
-- Name: items items_pkey; Type: CONSTRAINT; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.items
    ADD CONSTRAINT items_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: hacker_news; Owner: zer4bab
--

ALTER TABLE ONLY hacker_news.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: idx_items_by; Type: INDEX; Schema: hacker_news; Owner: zer4bab
--

CREATE INDEX idx_items_by ON hacker_news.items USING btree (by);


--
-- Name: idx_items_time; Type: INDEX; Schema: hacker_news; Owner: zer4bab
--

CREATE INDEX idx_items_time ON hacker_news.items USING btree ("time");


--
-- Name: idx_items_type; Type: INDEX; Schema: hacker_news; Owner: zer4bab
--

CREATE INDEX idx_items_type ON hacker_news.items USING btree (type);


--
-- Name: SCHEMA hacker_news; Type: ACL; Schema: -; Owner: zer4bab
--

GRANT USAGE ON SCHEMA hacker_news TO sy91dhb;


--
-- Name: TABLE items; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items TO sy91dhb;


--
-- Name: TABLE items_by_month; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month TO sy91dhb;


--
-- Name: TABLE items_by_month_2006_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2006_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2006_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2006_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2007_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2007_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2008_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2008_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2009_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2009_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2010_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2010_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2011_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2011_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2012_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2012_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2013_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2013_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2014_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2014_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2015_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2015_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2016_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2016_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2017_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2017_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2018_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2018_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2019_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2019_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2020_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2020_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2021_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2021_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2022_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2022_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_10 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_11; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_11 TO sy91dhb;


--
-- Name: TABLE items_by_month_2023_12; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2023_12 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_01; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_01 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_02; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_02 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_03; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_03 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_04; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_04 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_05; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_05 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_06; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_06 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_07; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_07 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_08; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_08 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_09; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_09 TO sy91dhb;


--
-- Name: TABLE items_by_month_2024_10; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_month_2024_10 TO sy91dhb;


--
-- Name: TABLE "items_by_month_YYYY_XX"; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news."items_by_month_YYYY_XX" TO sy91dhb;


--
-- Name: TABLE items_by_year; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year TO sy91dhb;


--
-- Name: TABLE items_by_year_2006; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2006 TO sy91dhb;


--
-- Name: TABLE items_by_year_2007; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2007 TO sy91dhb;


--
-- Name: TABLE items_by_year_2008; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2008 TO sy91dhb;


--
-- Name: TABLE items_by_year_2009; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2009 TO sy91dhb;


--
-- Name: TABLE items_by_year_2010; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2010 TO sy91dhb;


--
-- Name: TABLE items_by_year_2011; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2011 TO sy91dhb;


--
-- Name: TABLE items_by_year_2012; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2012 TO sy91dhb;


--
-- Name: TABLE items_by_year_2013; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2013 TO sy91dhb;


--
-- Name: TABLE items_by_year_2014; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2014 TO sy91dhb;


--
-- Name: TABLE items_by_year_2015; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2015 TO sy91dhb;


--
-- Name: TABLE items_by_year_2016; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2016 TO sy91dhb;


--
-- Name: TABLE items_by_year_2017; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2017 TO sy91dhb;


--
-- Name: TABLE items_by_year_2018; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2018 TO sy91dhb;


--
-- Name: TABLE items_by_year_2019; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2019 TO sy91dhb;


--
-- Name: TABLE items_by_year_2020; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2020 TO sy91dhb;


--
-- Name: TABLE items_by_year_2021; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2021 TO sy91dhb;


--
-- Name: TABLE items_by_year_2022; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2022 TO sy91dhb;


--
-- Name: TABLE items_by_year_2023; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2023 TO sy91dhb;


--
-- Name: TABLE items_by_year_2024; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.items_by_year_2024 TO sy91dhb;


--
-- Name: TABLE "items_by_year_YYYY"; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news."items_by_year_YYYY" TO sy91dhb;


--
-- Name: TABLE users; Type: ACL; Schema: hacker_news; Owner: zer4bab
--

GRANT SELECT ON TABLE hacker_news.users TO sy91dhb;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: hacker_news; Owner: zer4bab
--

ALTER DEFAULT PRIVILEGES FOR ROLE zer4bab IN SCHEMA hacker_news GRANT SELECT ON TABLES TO sy91dhb;


--
-- PostgreSQL database dump complete
--

