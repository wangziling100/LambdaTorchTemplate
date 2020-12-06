old_dir=`pwd`
cd $(dirname $0)
rm fill_template.sh
rm install_frep.sh
rm rename.sh
cd ..
rm autoCI.config.json.template
rm env.json.template
rm samconfig.toml.template
rm template.yaml.template
rm README.md
rm config.yaml
cd .github/workflows
sed -i '24,33d' autoCI.yaml
cd $old_dir