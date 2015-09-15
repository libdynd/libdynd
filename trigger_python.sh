declare exitCode
curl -sSL https://raw.githubusercontent.com/alrra/travis-after-all/1.4.1/lib/travis-after-all.js | node
exitCode=$?
#if [ $exitCode -eq 0 ]; then ./trigger_python.sh $TRAVIS_TOKEN; fi

body='{
"request": {
  "branch":"master"
}}'

curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Travis-API-Version: 3" \
  -H "Authorization: token $1" \
  -d "$body" \
  https://api.travis-ci.org/repo/libdynd%2Fdynd-python/requests
