package config

import (
	"fmt"

	"github.com/spf13/viper"
)

type Config struct {
	// Add tags to the TOP LEVEL fields too!
	Server struct {
		Port string `mapstructure:"port"`
	} `mapstructure:"server"`

	Grpc struct {
		PythonWorkerAddress string `mapstructure:"python_worker_address"`
		Timeout int    `mapstructure:"timeout_seconds"`
	} `mapstructure:"grpc"`

	Image struct {
		TileSize int `mapstructure:"tile_size"`
	} `mapstructure:"image"`

	Storage struct {
        Endpoint     string `mapstructure:"endpoint"`
        AccessKey    string `mapstructure:"access_key"`
        SecretKey    string `mapstructure:"secret_key"`
        UseSSL       bool   `mapstructure:"use_ssl"`
        BucketInput  string `mapstructure:"bucket_input"`
        BucketOutput string `mapstructure:"bucket_output"`
    } `mapstructure:"storage"`
}

func LoadConfig() (*Config, error) {
	viper.AddConfigPath(".")      // Look in current directory
	viper.SetConfigName("config") // Look for config.yaml
	viper.SetConfigType("yaml")

	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("config file not found: %w", err)
	}

	var cfg Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// --- DEBUG PRINT ---
	// This will show us immediately if the config loaded or if it's still empty
	fmt.Printf("[CONFIG] Loaded -> Port: %s | Grpc: %s | TileSize: %d\n", 
		cfg.Server.Port, cfg.Grpc.PythonWorkerAddress, cfg.Image.TileSize)
	// -------------------

	return &cfg, nil
}