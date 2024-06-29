val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "BankMarketing",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,  
      // https://mvnrepository.com/artifact/com.intel.analytics.bigdl.core/parent
      "com.intel.analytics.bigdl.core" % "parent" % "2.1.0" pomOnly(),
      "com.intel.analytics.bigdl.core.dist" % "linux64" % "2.1.0" pomOnly(),
      // "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
      // "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "2.0.0",
      // "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "2.0.0",
      // "com.github.scopt" %% "scopt" % "4.1.0",
      "com.google.protobuf" % "protobuf-java" % "3.22.2",
      "org.scala-lang.modules" %% "scala-collection-compat" % "2.8.0",      
    )

  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
