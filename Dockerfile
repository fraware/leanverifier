FROM python:3.9-slim AS base

# 1. Install needed packages, including `git`
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git wget xz-utils g++ libgmp-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# 2. Install elan (for Lean 4)
RUN curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y

# Ensure the correct PATH is set for elan and lake
ENV PATH="/root/.elan/bin:$PATH"

# 3. Install Lean 4 and lake explicitly
RUN elan toolchain install leanprover/lean4:4.17.0 && \
    elan default leanprover/lean4:4.17.0

# 4. Create and move to /app
WORKDIR /app

# 5. Copy entire project into /app
COPY . /app

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r translator/requirements.txt
RUN pip install --no-cache-dir flask
RUN pip install --no-cache-dir graphviz

# 7. Switch to your Lean project folder
WORKDIR /app/lean

# 8. Build with Lake (which will clone mathlib if needed)
RUN lake build

# 9. Expose port 5000 for Flask
EXPOSE 5000

# 10. Launch your Flask web app
CMD ["python", "/app/webapp/app.py"]
