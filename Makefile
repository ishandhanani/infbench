.PHONY: lint test setup-configs dashboard sync-to-cloud delete-from-cloud

NATS_VERSION ?= v2.10.28
ETCD_VERSION ?= v3.5.21
LOGS_DIR ?= logs

default:
	./run_dashboard.sh

lint:
	uvx pre-commit run --all-files

test:
	cd /Users/idhanani/Desktop/benchmarks/infbench && uv run python -m tests.test_basic && uv run python -m tests.test_aggregations

dashboard:
	uv run streamlit run dashboard/app.py

sync-to-cloud:
	@echo "☁️  Syncing benchmark results to cloud storage..."
	@echo "📁 Logs directory: $(LOGS_DIR)"
	@uv run python -m srtslurm.sync_results --logs-dir $(LOGS_DIR) push-all
	@echo "✅ Sync complete!"

delete-from-cloud:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "❌ Error: RUN_ID not specified"; \
		echo "Usage: make delete-from-cloud RUN_ID=3667_1P_1D_20251110_192145"; \
		exit 1; \
	fi
	@uv run python -m srtslurm.sync_results delete $(RUN_ID)

setup:
	@echo "📦 Setting up configs and logs directories..."
	@mkdir -p logs
	@ARCH=$$(uname -m); \
	case "$$ARCH" in \
		x86_64)  ARCH_SHORT="amd64" ;; \
		aarch64) ARCH_SHORT="arm64" ;; \
		*) echo "❌ Unsupported architecture: $$ARCH"; exit 1 ;; \
	esac; \
	echo "⬇️  Downloading Python wheels..."; \
	wget -q --show-progress -P configs https://files.pythonhosted.org/packages/dc/b7/62fb0edaeae0943731d0e1d3e1455b0a8a94ef448aa5bd8ffe33288ab464/ai_dynamo-0.6.1-py3-none-any.whl; \
	echo "⬇️  Downloading ai_dynamo_runtime for aarch64 (GB200)..."; \
	wget -q --show-progress -P configs https://files.pythonhosted.org/packages/b8/0c/076268db6ff2c87663a0d70f7ce7a6a1c566ac1383981d9c82437de2ff98/ai_dynamo_runtime-0.6.1-cp310-abi3-manylinux_2_28_aarch64.whl; \
	echo "⬇️  Downloading ai_dynamo_runtime for x86_64 (H100)..."; \
	wget -q --show-progress -P configs https://files.pythonhosted.org/packages/99/fc/cd7172407aeb07fc83fa94eb51281280847e5ec7fb3c6aedb1a02cf4e7ea/ai_dynamo_runtime-0.6.1-cp310-abi3-manylinux_2_28_x86_64.whl; \
	echo "⬇️  Downloading NATS ($(NATS_VERSION)) for $$ARCH_SHORT..."; \
	NATS_DEB="nats-server-$(NATS_VERSION)-$$ARCH_SHORT.deb"; \
	NATS_URL="https://github.com/nats-io/nats-server/releases/download/$(NATS_VERSION)/$$NATS_DEB"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$NATS_URL" -O "configs/$$NATS_DEB"; \
	echo "📁 Extracting NATS binary..."; \
	TMP_DIR=$$(mktemp -d); \
	dpkg-deb -x "configs/$$NATS_DEB" "$$TMP_DIR"; \
	if [ -f "$$TMP_DIR/usr/local/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/local/bin/nats-server" configs/; \
	elif [ -f "$$TMP_DIR/usr/bin/nats-server" ]; then \
		cp "$$TMP_DIR/usr/bin/nats-server" configs/; \
	else \
		echo "❌ Could not find nats-server binary inside the .deb package"; \
		ls -R "$$TMP_DIR" | head -n 50; \
		exit 1; \
	fi; \
	chmod +x configs/nats-server; \
	rm -rf "$$TMP_DIR" "configs/$$NATS_DEB"; \
	echo "⬇️  Downloading ETCD ($(ETCD_VERSION)) for $$ARCH_SHORT..."; \
	ETCD_TAR="etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT.tar.gz"; \
	ETCD_URL="https://github.com/etcd-io/etcd/releases/download/$(ETCD_VERSION)/$$ETCD_TAR"; \
	wget -q --show-progress --tries=3 --waitretry=5 "$$ETCD_URL" -O "configs/$$ETCD_TAR"; \
	echo "📁 Extracting ETCD binaries..."; \
	tar -xzf "configs/$$ETCD_TAR" --strip-components=1 -C configs etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcd etcd-$(ETCD_VERSION)-linux-$$ARCH_SHORT/etcdctl; \
	chmod +x configs/etcd configs/etcdctl; \
	rm "configs/$$ETCD_TAR"; \
	echo "✅ Done. Contents of configs directory:"; \
	ls -lh configs/; \
	echo ""; \
	echo "⚙️  Setting up srtslurm.yaml..."; \
	if [ -f srtslurm.yaml ]; then \
		echo "ℹ️  srtslurm.yaml already exists, skipping..."; \
	else \
		echo "Creating srtslurm.yaml with your cluster settings..."; \
		echo ""; \
		read -p "Enter SLURM account [restricted]: " account; \
		account=$${account:-restricted}; \
		read -p "Enter SLURM partition [batch]: " partition; \
		partition=$${partition:-batch}; \
		read -p "Enter network interface [enP6p9s0np0]: " network; \
		network=$${network:-enP6p9s0np0}; \
		read -p "Enter GPUs per node [8]: " gpus_per_node; \
		gpus_per_node=$${gpus_per_node:-8}; \
		read -p "Enter time limit [4:00:00]: " time_limit; \
		time_limit=$${time_limit:-4:00:00}; \
		read -p "Enter container image path (optional): " container; \
		container=$${container:-}; \
		echo ""; \
		echo "# SRT SLURM Configuration" > srtslurm.yaml; \
		echo "" >> srtslurm.yaml; \
		echo "cluster:" >> srtslurm.yaml; \
		echo "  account: \"$$account\"" >> srtslurm.yaml; \
		echo "  partition: \"$$partition\"" >> srtslurm.yaml; \
		echo "  network_interface: \"$$network\"" >> srtslurm.yaml; \
		echo "  gpus_per_node: $$gpus_per_node" >> srtslurm.yaml; \
		echo "  time_limit: \"$$time_limit\"" >> srtslurm.yaml; \
		echo "  container_image: \"$$container\"" >> srtslurm.yaml; \
		echo "" >> srtslurm.yaml; \
		echo "cloud:" >> srtslurm.yaml; \
		echo "  endpoint_url: \"\"" >> srtslurm.yaml; \
		echo "  bucket: \"\"" >> srtslurm.yaml; \
		echo "  prefix: \"benchmark-results/\"" >> srtslurm.yaml; \
		echo "✅ Created srtslurm.yaml"; \
		echo "   You can edit it anytime or run: cp srtslurm.yaml.example srtslurm.yaml"; \
	fi