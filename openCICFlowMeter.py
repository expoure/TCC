import subprocess

def runCICFlowMeter():
    subprocess.call(['/usr/lib/jvm/java-8-openjdk/jre/bin/java', '-jar', 'CICFlowMeterV3-0.0.4-SNAPSHOT.jar'])
