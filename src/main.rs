mod config;
mod discord_handler;
mod tokenizer;

use std::sync::Arc;

use anyhow::Result;
use config::BotConfig;
use discord_handler::DiscordHandler;
use twilight_gateway::{Event, EventTypeFlags, Intents, Shard, ShardId, StreamExt as _};
use twilight_http::Client as HttpClient;
use twilight_model::{
    application::{
        command::{CommandOption, CommandOptionType},
        interaction::{InteractionData, InteractionType, application_command::CommandOptionValue},
    },
    channel::{ChannelType, message::MessageFlags},
    guild::Permissions,
    http::interaction::{InteractionResponse, InteractionResponseData, InteractionResponseType},
    id::{
        Id,
        marker::{ApplicationMarker, ChannelMarker},
    },
};

fn main() -> Result<()> {
    install_rustls_provider();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let config = BotConfig::from_env()?;

    let http = Arc::new(HttpClient::new(config.discord_token().to_owned()));
    let current_user_id = http.current_user().await?.model().await?.id;
    let application_id = http.current_user_application().await?.model().await?.id;
    let handler = DiscordHandler::new(config.clone(), current_user_id).await?;
    register_slash_commands(&http, application_id).await?;

    let intents = Intents::GUILDS | Intents::GUILD_MESSAGES | Intents::MESSAGE_CONTENT;
    let mut shard = Shard::new(ShardId::ONE, config.discord_token().to_owned(), intents);

    println!(
        "Bot started. target_channel_id=unset(use /set_channel), cooldown={}s, generation<= {} words, temp={}, min_words_before_eos={}, storage_min_edge_count={}, storage_compression={}",
        config.reply_cooldown_secs(),
        config.max_words().get(),
        config.temperature().get(),
        config.min_words_before_eos().get(),
        config.storage_min_edge_count(),
        config.storage_compression().as_env_value(),
    );

    while let Some(item) = shard.next_event(EventTypeFlags::all()).await {
        let event = match item {
            Ok(event) => event,
            Err(source) => {
                eprintln!("Gateway event receive error: {source}");
                continue;
            }
        };

        if let Event::MessageCreate(message) = event {
            if let Err(error) = handler
                .handle_message(
                    &http,
                    message.channel_id,
                    message.author.id,
                    message.author.bot,
                    &message.content,
                )
                .await
            {
                eprintln!("Failed to process message: {error}");
            }
            continue;
        }

        if let Event::InteractionCreate(interaction) = event
            && let Err(error) =
                handle_interaction_command(&http, &handler, interaction.0, application_id).await
        {
            eprintln!("Failed to process interaction: {error}");
        }
    }

    Ok(())
}

fn install_rustls_provider() {
    if let Err(_already_installed) = rustls::crypto::aws_lc_rs::default_provider().install_default()
    {
        // 既に別経路でProviderが設定済みの場合はそのまま利用する。
    }
}

async fn register_slash_commands(
    http: &HttpClient,
    application_id: Id<ApplicationMarker>,
) -> Result<()> {
    let command_options = [CommandOption {
        autocomplete: None,
        channel_types: Some(vec![
            ChannelType::GuildText,
            ChannelType::GuildAnnouncement,
            ChannelType::AnnouncementThread,
            ChannelType::PublicThread,
            ChannelType::PrivateThread,
        ]),
        choices: None,
        description: "学習と返信の対象チャンネル".to_owned(),
        description_localizations: None,
        kind: CommandOptionType::Channel,
        max_length: None,
        max_value: None,
        min_length: None,
        min_value: None,
        name: "channel".to_owned(),
        name_localizations: None,
        options: None,
        required: Some(true),
    }];

    let _ = http
        .interaction(application_id)
        .create_global_command()
        .chat_input("set_channel", "学習・応答対象チャンネルを設定します")
        .default_member_permissions(Permissions::MANAGE_CHANNELS)
        .dm_permission(false)
        .command_options(&command_options)
        .await?;

    Ok(())
}

async fn handle_interaction_command(
    http: &HttpClient,
    handler: &DiscordHandler,
    interaction: twilight_model::application::interaction::Interaction,
    application_id: Id<ApplicationMarker>,
) -> Result<()> {
    if interaction.kind != InteractionType::ApplicationCommand {
        return Ok(());
    }

    let Some(InteractionData::ApplicationCommand(command_data)) = interaction.data.as_ref() else {
        return Ok(());
    };

    if command_data.name != "set_channel" {
        return Ok(());
    }

    let response_message =
        if let Some(channel_id) = extract_channel_option(command_data.options.as_slice()) {
            handler.set_target_channel(channel_id).await;

            format!("対象チャンネルを <#{channel_id}> に設定しました。")
        } else {
            "チャンネル指定を解釈できませんでした。もう一度実行してください。".to_owned()
        };

    let response = InteractionResponse {
        kind: InteractionResponseType::ChannelMessageWithSource,
        data: Some(InteractionResponseData {
            content: Some(response_message),
            flags: Some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        }),
    };

    let _ = http
        .interaction(application_id)
        .create_response(interaction.id, &interaction.token, &response)
        .await?;

    Ok(())
}

fn extract_channel_option(
    options: &[twilight_model::application::interaction::application_command::CommandDataOption],
) -> Option<Id<ChannelMarker>> {
    options.iter().find_map(|option| {
        if option.name != "channel" {
            return None;
        }

        if let CommandOptionValue::Channel(channel_id) = &option.value {
            Some(*channel_id)
        } else {
            None
        }
    })
}
