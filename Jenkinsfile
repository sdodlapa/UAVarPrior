properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '15'))
])

node("docker") { stage('build') { timeout(time: 1, unit: 'HOURS') {
  checkout scm
  def commit = sh(returnStdout: true, script: "git rev-parse HEAD").trim()
  def workDir = pwd()
  def tmpDir = pwd(tmp:true)
  def img = docker.build("uavarprior/uavarprior:${env.BRANCH_NAME}", ".")
  img.inside() {
    sh '''#!/bin/bash -ex
      pip install -e ".[dev]"
      export PYTHONPATH=$PWD/src
      pytest tests/
      make -C docs html
    '''
  }

  dir("$tmpDir/gh-pages") {
    def subdir = env.BRANCH_NAME
    git(url: "ssh://git@github.com/SanjeevaRDodlapati/UAVarPrior.git", branch: "gh-pages", credentialsId: "ssh", changelog: false)
    sh "mkdir $subdir || rm -rf $subdir/[[:lower:]_]*"
    sh "mv $workDir/docs/build/html/* $subdir"
    sh "git add -A $subdir"
    sh """
      git commit --author='Flatiron Jenkins <jenkins@flatironinstitute.org>' --allow-empty -m 'Generated documentation' -m '${env.BUILD_TAG} ${commit}'
    """
    sh "git push origin gh-pages"
  }
} } }
