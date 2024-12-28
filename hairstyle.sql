# HeidiSQL Dump 
#
# --------------------------------------------------------
# Host:                 127.0.0.1
# Database:             hairstyle
# Server version:       5.4.3-beta-community
# Server OS:            Win32
# Target-Compatibility: Standard ANSI SQL
# HeidiSQL version:     3.1 RC1 Revision: 1064
# --------------------------------------------------------

/*!40100 SET CHARACTER SET latin1;*/
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ANSI';*/
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;*/


#
# Database structure for database 'hairstyle'
#

CREATE DATABASE /*!32312 IF NOT EXISTS*/ "hairstyle" /*!40100 DEFAULT CHARACTER SET latin1 */;

USE "hairstyle";


#
# Table structure for table 'user_information'
#

CREATE TABLE /*!32312 IF NOT EXISTS*/ "user_information" (
  "userid" int(11) NOT NULL AUTO_INCREMENT,
  "username" varchar(50) DEFAULT NULL,
  "password" varchar(50) DEFAULT NULL,
  "address" varchar(50) DEFAULT NULL,
  "emailid" varchar(50) DEFAULT NULL,
  "mobile" varchar(50) DEFAULT NULL,
  PRIMARY KEY ("userid")
) AUTO_INCREMENT=5 /*!40100 DEFAULT CHARSET=latin1*/;



#
# Dumping data for table 'user_information'
#

/*!40000 ALTER TABLE "user_information" DISABLE KEYS;*/
LOCK TABLES "user_information" WRITE;
REPLACE INTO "user_information" ("userid", "username", "password", "address", "emailid", "mobile") VALUES
	(1,'priya','priya','cbe','priya@gmail.com','9856325698');
REPLACE INTO "user_information" ("userid", "username", "password", "address", "emailid", "mobile") VALUES
	(4,'afd','ds','sfd','df','df');
UNLOCK TABLES;
/*!40000 ALTER TABLE "user_information" ENABLE KEYS;*/
/*!40101 SET SQL_MODE=@OLD_SQL_MODE;*/
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;*/
