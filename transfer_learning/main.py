from robustness import train, attacker
import tools



if __name__ == "__main__":
    parser = tools.make_parser()
    args = parser.parse_args()
    args = tools.check_and_fill_config_training_and_model_loader_args(args)
    tools.check_that_dataset_supports_per_class_accuracy(args)

    store = tools.make_store(args)

    ds, train_loader, validation_loader = tools.get_dataset_and_loaders(args)

    args = tools.modify_args_if_using_per_class_accuracy(args, validation_loader)

    model = tools.load_model_from_id(args.model_id)
    model = tools.replace_fc_layer(model, ds.num_classes)
    model = attacker.AttackerModel(model, ds)

    fine_tuning_params = tools.get_fine_tuning_params(model, freeze_level=args.freeze_level)

    train.train_model(args, model, (train_loader, validation_loader), store=store,
                      checkpoint=None, update_params=fine_tuning_params)
