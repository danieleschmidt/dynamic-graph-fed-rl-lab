# Terraform configuration for Dynamic Graph Fed-RL infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  # Configure remote state storage
  backend "s3" {
    # Configure these variables in terraform.tfvars or environment
    bucket = var.terraform_state_bucket
    key    = "dynamic-graph-fed-rl/terraform.tfstate"
    region = var.aws_region
    
    # Enable state locking
    dynamodb_table = var.terraform_lock_table
    encrypt        = true
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "dynamic-graph-fed-rl"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Local variables
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC for the EKS cluster
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${local.cluster_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Enable VPC CNI
  enable_ipv6 = false
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  })
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = "1"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Managed node groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      instance_types = var.general_node_instance_types
      
      min_size     = var.general_node_min_size
      max_size     = var.general_node_max_size
      desired_size = var.general_node_desired_size
      
      labels = {
        workload = "general"
      }
      
      taints = []
    }
    
    # GPU nodes for training
    gpu = {
      instance_types = var.gpu_node_instance_types
      
      min_size     = var.gpu_node_min_size
      max_size     = var.gpu_node_max_size
      desired_size = var.gpu_node_desired_size
      
      labels = {
        workload = "gpu-training"
        "nvidia.com/gpu" = "true"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      # Install NVIDIA device plugin
      pre_bootstrap_user_data = <<-EOT
        /etc/eks/bootstrap.sh ${local.cluster_name}
        /opt/aws/bin/cfn-signal --exit-code $? --stack  ${var.cloudformation_stack_name} --resource NodeGroup --region ${var.aws_region}
      EOT
    }
  }
  
  tags = local.common_tags
}