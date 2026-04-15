import torch


def train_epoch_mscl(encoder, projector1, projector2, train_loader, train_transform, criterion, optimizer, scheduler, metatradeoff, device):
    epoch_loss = 0.0

    for data, target, metadata_target in train_loader:
        data, target, metadata_target = data.to(device), target.to(device), metadata_target.to(device)

        with torch.no_grad():
            data_t1 = train_transform(data)
            data_t2 = train_transform(data)

        feat1, feat2 = encoder(data_t1), encoder(data_t2)
        proj1, proj2 = projector1(feat1), projector1(feat2)
        proj1_prime, proj2_prime = projector2(feat1), projector2(feat2)

        loss = (1. - metatradeoff) * criterion(proj1, proj2, metadata_target) + metatradeoff * criterion(proj1_prime, proj2_prime, target)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    scheduler.step()

    return epoch_loss


def train_mscl(encoder, projector1, projector2, train_loader, train_transform, criterion, optimizer, scheduler, epochs, metatradeoff, device):
    best_loss = None
    train_losses = []

    encoder.train()
    projector1.train()
    projector2.train()

    for i in range(1, epochs + 1):
        print(f"Epoch {i}")
        train_loss = train_epoch_mscl(encoder, projector1, projector2, train_loader, train_transform, criterion, optimizer, scheduler, metatradeoff, device)
        print(f"Current Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        if best_loss is None or train_loss < best_loss:
            best_loss = train_loss

    final_state = {"encoder": encoder.state_dict()}
    print(f"Last Loss: {train_loss:.4f}\tBest Loss: {best_loss:.4f}")

    return train_losses, encoder, final_state


def linear_train_epoch(encoder, classifier, train_loader, val_transform, criterion, optimizer, device, num_classes=4):
    epoch_loss = 0.0
    TP = [0] * num_classes
    GT = [0] * num_classes

    classifier.train()

    for data, target, _ in train_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            features = encoder(val_transform(data))

        optimizer.zero_grad()
        output = classifier(features)
        loss = criterion(output, target)
        epoch_loss += loss.item()

        _, labels_predicted = torch.max(output, dim=1)
        for idx in range(num_classes):
            TP[idx] += torch.logical_and((labels_predicted == idx), (target == idx)).sum().item()
            GT[idx] += (target == idx).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    se = sum(TP[1:]) / sum(GT[1:])
    sp = TP[0] / GT[0]
    icbhi_score = (se + sp) / 2
    acc = sum(TP) / sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc


def linear_eval_epoch(encoder, classifier, val_loader, val_transform, criterion, device, num_classes=4):
    epoch_loss = 0.0
    TP = [0] * num_classes
    GT = [0] * num_classes

    classifier.eval()
    encoder.eval()

    with torch.no_grad():
        for data, target, _ in val_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(encoder(val_transform(data)))
            loss = criterion(output, target)
            epoch_loss += loss.item()

            _, labels_predicted = torch.max(output, dim=1)
            for idx in range(num_classes):
                TP[idx] += torch.logical_and((labels_predicted == idx), (target == idx)).sum().item()
                GT[idx] += (target == idx).sum().item()

    epoch_loss = epoch_loss / len(val_loader)
    se = sum(TP[1:]) / sum(GT[1:])
    sp = TP[0] / GT[0]
    icbhi_score = (se + sp) / 2
    acc = sum(TP) / sum(GT)

    return epoch_loss, se, sp, icbhi_score, acc


def linear_mscl(encoder, checkpoint, classifier, train_loader, val_loader, val_transform, criterion, optimizer, epochs, device, num_classes=4):
    best_icbhi_score = 0
    best_se = 0
    best_sp = 0
    best_epoch_icbhi = 0
    best_state = None

    state_dict = checkpoint["encoder"]
    encoder.load_state_dict(state_dict)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    for i in range(1, epochs + 1):
        print(f"Epoch {i}")

        train_loss, train_se, train_sp, train_icbhi_score, train_acc = linear_train_epoch(
            encoder, classifier, train_loader, val_transform, criterion, optimizer, device, num_classes)
        print(f"Train loss: {train_loss:.4f}\tSE: {train_se:.4f}\tSP: {train_sp:.4f}\tScore: {train_icbhi_score:.4f}\tAcc: {train_acc:.4f}")

        val_loss, val_se, val_sp, val_icbhi_score, val_acc = linear_eval_epoch(
            encoder, classifier, val_loader, val_transform, criterion, device, num_classes)
        print(f"Val loss: {val_loss:.4f}\tSE: {val_se:.4f}\tSP: {val_sp:.4f}\tScore: {val_icbhi_score:.4f}\tAcc: {val_acc:.4f}")

        if i == 1 or val_icbhi_score > best_icbhi_score:
            best_epoch_icbhi = i
            best_icbhi_score = val_icbhi_score
            best_se = val_se
            best_sp = val_sp
            best_state = {"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}

    print(f"Best score: {best_icbhi_score:.4f} (SE:{best_se:.4f} SP:{best_sp:.4f}) at epoch {best_epoch_icbhi}")

    return best_state, best_icbhi_score, best_se, best_sp
