declare exitCode
curl -sSL https://raw.githubusercontent.com/alrra/travis-after-all/1.4.1/lib/travis-after-all.js | node
exitCode=$?

if [ $exitCode -ne 0 ]; then exit 0; fi

if [ $TRAVIS_BRANCH != "master" ] || [ $TRAVIS_PULL_REQUEST != "false" ]; then
  exit 0;
fi

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
